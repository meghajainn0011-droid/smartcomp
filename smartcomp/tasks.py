"""One-call desktop tasks: plan, queue, execute, report, learn.

The agent's complaint this answers: a simple flow ("open firefox, visit the
page, click Profile, tell me what it says") used to cost 6-10 MCP calls, each
one a full model turn and a fresh perception pass, so a 5-second job took
minutes and a lot of tokens. Here the whole flow is ONE tool call. The steps
run inside the daemon, back to back, on the display the daemon is already
attached to, and the caller gets one small JSON report.

Three properties matter:

* **One report.** `answer` is the readback (window title + visible lines), and
  `steps` says what happened in order. Success and failure both come back in
  the same shape, so the agent can decide its next move from one response.
* **A queue.** Two agents sending tasks do not fight over the pointer: one
  worker executes them in arrival order, and the second call waits. Tasks are
  cancellable between steps.
* **It learns.** A task that worked is saved as a plan keyed by its goal, so
  the same request later skips planning entirely (`learned: true`); a small
  word-level router (`policy.py`) picks the plan shape for new phrasings.

The planner is deliberately not an LLM. It understands explicit arguments
(`app`, `url`, `click`, `expect`) and the imperative verbs an agent actually
uses; anything else comes back as a clear "I need a url/label" error instead
of a wrong click. `steps=` takes a literal plan for callers that want full
control without six round trips.
"""

from __future__ import annotations

import queue
import re
import threading
import time
import uuid

from . import config
from .policy import Router

BROWSERS = ("firefox", "chromium", "brave")
DEFAULT_BROWSER = "firefox"
MAX_QUEUE = 8
STOP_WORDS = {"the", "a", "an", "to", "on", "in", "of", "and", "then", "please",
              "my", "me", "for", "with", "it", "is", "at", "by", "from"}


# -- goals -> plan -----------------------------------------------------------

def norm_goal(goal: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (goal or "").lower()))


def _words(text: str) -> set[str]:
    return {w for w in norm_goal(text).split() if w not in STOP_WORDS}


def jaccard(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


URL_RE = re.compile(r"(https?://[^\s,;]+|www\.[^\s,;]+|"
                    r"\b[\w-]+\.(?:com|org|net|io|in|co|ai|dev|app|me)\b[^\s,;]*)")
CLICK_RE = re.compile(r"\b(?:click|press|tap|hit)\s+(?:on\s+)?(?:the\s+)?"
                      r"(?:[\"']([^\"']+)[\"']|([\w .:+-]{1,40}?))"
                      r"(?=\s+(?:and|then|button|link|tab|icon)\b|[,;.]|$)", re.I)
TYPE_RE = re.compile(r"\b(?:type|enter|search for|search)\s+[\"']?([^\"';]+)", re.I)


def parse_goal(goal: str, app: str | None = None, url: str | None = None,
               click: str | None = None, expect: str | None = None) -> dict:
    """The plan for a goal, or a precise question when a piece is missing."""
    text = goal or ""
    found_url = URL_RE.search(text)
    url = url or (found_url.group(0) if found_url else None)
    if click is None:
        m = CLICK_RE.search(text)
        if m:
            click = (m.group(1) or m.group(2) or "").strip(" .")
    if app is None:
        low = text.lower()
        app = next((name for name in ("firefox", "chromium", "brave", "terminal",
                                      "text-editor", "file-manager", "image-viewer")
                    if name in low), None)
    if not app and (url or click):
        app = DEFAULT_BROWSER
    expect = expect or click

    if url:
        plan = [{"tool": "launch_app", "args": {"name": app or DEFAULT_BROWSER}},
                {"tool": "goto", "args": {"url": url}}]
        if click:
            plan.append({"tool": "wait_text", "args": {"text": click, "timeout_s": 20}})
            plan.append({"tool": "click_label", "args": {"label": click, "timeout_s": 6}})
        elif expect:
            plan.append({"tool": "wait_text", "args": {"text": expect, "timeout_s": 20}})
        plan.append({"tool": "settle", "args": {"quiet_s": 0.6, "timeout_s": 4}})
        plan.append({"tool": "report", "args": {}})
        return {"ok": True, "template": "goto", "plan": plan}
    if click:
        plan = [{"tool": "launch_app", "args": {"name": app or DEFAULT_BROWSER}},
                {"tool": "wait_text", "args": {"text": click, "timeout_s": 10}},
                {"tool": "click_label", "args": {"label": click, "timeout_s": 6}},
                {"tool": "settle", "args": {"quiet_s": 0.5, "timeout_s": 3}},
                {"tool": "report", "args": {}}]
        return {"ok": True, "template": "click", "plan": plan}
    typed = TYPE_RE.search(text)
    if typed:
        plan = []
        if app:
            plan.append({"tool": "launch_app", "args": {"name": app}})
        plan += [{"tool": "type_text", "args": {"text": typed.group(1).strip()}},
                 {"tool": "report", "args": {}}]
        return {"ok": True, "template": "click", "plan": plan}
    if app:
        return {"ok": True, "template": "open",
                "plan": [{"tool": "launch_app", "args": {"name": app}},
                         {"tool": "report", "args": {}}]}
    return {"ok": True, "template": "read", "plan": [{"tool": "report", "args": {}}]}


def _plan_ok(plan: list[dict]) -> list[dict]:
    return [dict(s) for s in plan if isinstance(s, dict) and s.get("tool")]


# -- execution ---------------------------------------------------------------

class Task:
    def __init__(self, goal: str, plan: list[dict], template: str) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.goal = goal
        self.plan = plan
        self.template = template
        self.status = "queued"
        self.created = time.time()
        self.started: float | None = None
        self.ended: float | None = None
        self.result: dict | None = None
        self.events: list[dict] = []
        self.window: str | None = None
        self.cancel = threading.Event()
        self.done = threading.Event()

    def snapshot(self) -> dict:
        return {"task_id": self.id, "goal": self.goal, "template": self.template,
                "status": self.status, "created": self.created,
                "started": self.started, "ended": self.ended,
                "steps": list(self.events),
                "result": self.result}


class Runner:
    """One worker, arrival order, cancellable between steps."""

    def __init__(self, registry: dict, memory, router: Router | None = None) -> None:
        self.registry = registry
        self.memory = memory
        self.router = router or Router(config.state_dir() / "policy.json")
        self._queue: queue.Queue[Task] = queue.Queue()
        self._tasks: dict[str, Task] = {}
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None

    # -- public -----------------------------------------------------------
    def submit(self, goal: str, plan: list[dict], template: str,
               wait_s: float = 90.0) -> dict:
        task = Task(goal, plan, template)
        with self._lock:
            if self._queue.qsize() >= MAX_QUEUE:
                return {"ok": False, "error": "task queue is full", "busy": True,
                        "queued": self._queue.qsize()}
            self._tasks[task.id] = task
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._work, daemon=True,
                                                name="smartcomp-tasks")
                self._worker.start()
            self._queue.put(task)
            ahead = max(0, self._queue.qsize() - 1)
        if task.done.wait(max(0.1, wait_s)):
            return task.result or {"ok": False, "error": "task ended without a result"}
        return {"ok": False, "pending": True, "task_id": task.id,
                "queued_ahead": ahead,
                "hint": "still queued or running; call task_status(task_id)"}

    def cancel(self, task_id: str) -> dict:
        with self._lock:
            task = self._tasks.get(task_id)
        if task is None:
            return {"ok": False, "error": f"unknown task {task_id}"}
        task.cancel.set()
        return {"ok": True, "task_id": task_id, "status": task.status,
                "note": "cancelled between steps; a running step finishes first"}

    def snapshot(self, task_id: str | None = None) -> dict:
        with self._lock:
            if task_id:
                task = self._tasks.get(task_id)
                return {"ok": True, **task.snapshot()} if task else {
                    "ok": False, "error": f"unknown task {task_id}"}
            tasks = list(self._tasks.values())
        return {"ok": True, "queued": self._queue.qsize(),
                "running": sum(1 for t in tasks if t.status == "running"),
                "tasks": [t.snapshot() for t in tasks[-10:]],
                "model": self.router.status()}

    # -- worker -----------------------------------------------------------
    def _work(self) -> None:
        while True:
            task = self._queue.get()
            if task.cancel.is_set():
                task.status = "cancelled"
                task.ended = time.time()
                task.result = {"ok": False, "cancelled": True, "task_id": task.id,
                               "goal": task.goal}
                task.done.set()
                continue
            task.status = "running"
            task.started = time.time()
            try:
                task.result = self._run(task)
            except Exception as exc:  # a task must never kill the worker
                task.result = {"ok": False, "task_id": task.id, "goal": task.goal,
                               "error": f"{type(exc).__name__}: {exc}"}
            task.ended = time.time()
            task.status = "done" if task.result.get("ok") else "failed"
            # Learn BEFORE releasing the caller: a task that returns ok must be
            # replayable by the very next call, not "eventually".
            self._learn(task)
            task.done.set()

    def _learn(self, task: Task) -> None:
        ok = bool(task.result and task.result.get("ok"))
        try:
            self.router.observe(task.goal, task.template, ok)
        except Exception:
            pass
        if not ok or task.template in ("learned", "custom"):
            return
        name = f"task/{norm_goal(task.goal)[:60].replace(' ', '-') or 'task'}"
        try:
            self.memory.learn_skill(name, task.plan, f"task:{norm_goal(task.goal)}")
        except Exception:
            pass

    # -- steps ------------------------------------------------------------
    def _call(self, name: str, args: dict) -> dict:
        fn = self.registry.get(name)
        if fn is None:
            return {"ok": False, "error": f"unknown step {name!r}"}
        out = fn(**args)
        return out if isinstance(out, dict) else {"ok": bool(out), "result": out}

    @staticmethod
    def _ok(out: dict) -> bool:
        return not (out.get("ok") is False or out.get("error"))

    def _run(self, task: Task) -> dict:
        t0 = time.time()
        results: list[dict] = []
        answer: dict = {}
        for i, step in enumerate(task.plan):
            if task.cancel.is_set():
                return {"ok": False, "cancelled": True, "task_id": task.id,
                        "goal": task.goal, "steps": results}
            name, args = step["tool"], dict(step.get("args") or {})
            st = time.time()
            try:
                out = self._exec(task, name, args)
            except Exception as exc:
                out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            ok = self._ok(out)
            event = {"step": name, "ok": ok, "ms": round((time.time() - st) * 1000),
                     "detail": _detail(out)}
            results.append(event)
            task.events.append(event)
            if not ok and not step.get("optional"):
                return {"ok": False, "task_id": task.id, "goal": task.goal,
                        "template": task.template, "failed_at": name,
                        "error": out.get("error") or "step failed",
                        "hint": out.get("hint"),
                        "visible_text": out.get("visible_text"),
                        "matches": out.get("matches"),
                        "steps": results,
                        "ms": round((time.time() - t0) * 1000)}
            if name == "report" and isinstance(out.get("answer"), dict):
                answer = out["answer"]
        return {"ok": True, "task_id": task.id, "goal": task.goal,
                "template": task.template, "steps": results, "answer": answer,
                "ms": round((time.time() - t0) * 1000)}

    def _exec(self, task: Task, name: str, args: dict) -> dict:
        if name == "goto":
            return self._goto(task, args.get("url", ""), args.get("app"))
        if name == "wait_text":
            return self._wait_text(task, args.get("text", ""),
                                   float(args.get("timeout_s", 8)))
        if name == "click_label":
            return self._click_label(task, args.get("label", ""),
                                     float(args.get("timeout_s", 5)))
        if name == "settle":
            return self._settle(task, float(args.get("quiet_s", 0.5)),
                                float(args.get("timeout_s", 4)))
        if name == "report":
            return self._report(task, args.get("window_id"))
        return self._call(name, args)

    # -- virtual steps ----------------------------------------------------
    def _window_for(self, app: str) -> dict | None:
        out = self._call("list_windows", {})
        wins = out.get("windows") or []
        low = app.lower()
        aliases = {"text-editor": ("mousepad", "gedit", "kate"),
                   "file-manager": ("thunar", "nautilus", "dolphin"),
                   "terminal": ("qterminal", "xterm", "konsole", "alacritty")}
        names = (low,) + aliases.get(low, ())
        return next((w for w in wins
                     if any(n in (w.get("comm") or "").lower()
                            or n in (w.get("cmd") or "").lower()
                            or n in (w.get("instance") or "").lower()
                            or n in (w.get("class") or "").lower()
                            for n in names)), None)

    def _ensure_window(self, task: Task,
                       app: str) -> tuple[dict | None, dict | None]:
        win = self._window_for(app)
        if win:
            out = self._call("focus_window", {"window_id": win["id"]})
            task.window = win.get("id") or task.window
            return win, None if self._ok(out) else out
        out = self._call("launch_app", {"name": app, "report": False})
        if not self._ok(out):
            return None, out
        wid = out.get("window")
        title = out.get("title", "")
        if not wid:
            # A cold browser can take many seconds to map its first window;
            # launch_app only waits a few, so keep polling before giving up.
            deadline = time.time() + 15.0
            while time.time() < deadline:
                win = self._window_for(app)
                if win:
                    wid, title = win["id"], win.get("title", "")
                    break
                time.sleep(0.25)
        if not wid:
            return None, {"ok": False, "error": f"{app} launched without a window",
                          "hint": "app_status shows what is running"}
        task.window = wid
        return {"id": wid, "title": title, "resumed": out.get("resumed")}, None

    def _goto(self, task: Task, url: str, app: str | None = None) -> dict:
        if not url:
            return {"ok": False, "error": "goto needs a url"}
        app = app or BROWSERS[0]
        win, err = self._ensure_window(task, app)
        if err:
            return err
        self._call("key", {"combo": "ctrl+l", "report": False})
        typed = self._call("type_text", {"text": url, "report": False})
        if not self._ok(typed):
            return typed
        self._call("key", {"combo": "Return", "report": False})
        return {"ok": True, "url": url, "window": win.get("id") if win else None}

    def _wait_text(self, task: Task, text: str, timeout_s: float) -> dict:
        if not text:
            return {"ok": True, "skipped": "no text"}
        self._call("get_changes", {"detect": False})  # find reads the store
        deadline = time.time() + max(0.5, timeout_s)
        last: dict = {"count": 0}
        while time.time() < deadline and not task.cancel.is_set():
            last = self._call("find", {"text": text})
            if (last.get("count") or 0) > 0:
                return {"ok": True, "text": text, "matches": last.get("matches")}
            self._call("get_changes", {"detect": False})
            time.sleep(0.25)
        return {"ok": False, "error": f"{text!r} did not appear in {timeout_s:.0f}s",
                "hint": "the page may still be loading; increase timeout or check the label",
                "visible_text": self._visible_text()}

    def _click_label(self, task: Task, label: str, timeout_s: float) -> dict:
        if not label:
            return {"ok": False, "error": "click_label needs a label"}
        self._call("get_changes", {"detect": False})  # click resolves labels
        deadline = time.time() + max(0.5, timeout_s)
        last: dict = {}
        while time.time() < deadline and not task.cancel.is_set():
            last = self._call("click_element", {"label": label, "report": False})
            if self._ok(last):
                return last
            err = str(last.get("error") or "")
            if "elements match" in err:  # never guess between candidates
                return last
            self._call("get_changes", {"detect": False})
            time.sleep(0.25)
        return last or {"ok": False, "error": f"could not click {label!r}"}

    def _settle(self, task: Task, quiet_s: float, timeout_s: float) -> dict:
        """Wait for the page to stop: its title stops changing, cheaply.

        wait_for_change re-OCRs the screen on every poll and took 6-9 s on a
        freshly loaded page, which was most of a task's latency. A window
        title comes from list_windows (~0.1 s) and moves while the page loads.
        """
        deadline = time.time() + max(0.2, timeout_s)
        quiet_since: float | None = None
        last: str | None = None
        while time.time() < deadline and not task.cancel.is_set():
            title = None
            if task.window:
                wins = self._call("list_windows", {}).get("windows") or []
                title = next((w.get("title") for w in wins
                              if w.get("id") == task.window), None)
            if title is None:
                break
            if title == last:
                if quiet_since is not None and time.time() - quiet_since >= quiet_s:
                    return {"ok": True, "quiet": True, "title": title}
            else:
                quiet_since = time.time()
            last = title
            time.sleep(0.2)
        if last is None:  # no window to watch: a short pause is all we can do
            time.sleep(min(max(0.2, quiet_s), max(0.2, timeout_s)))
            return {"ok": True, "quiet": True, "via": "sleep"}
        return {"ok": True, "quiet": False, "note": "deadline reached", "title": last}

    def _visible_text(self, limit: int = 8) -> list[str]:
        out = self._call("describe_screen", {})
        return [(c.get("t") or "")[:60] for c in (out.get("controls") or [])
                if c.get("t")][:limit]

    def _report(self, task: Task | None = None,
                window_id: str | None = None) -> dict:
        wins = self._call("list_windows", {}).get("windows") or []
        window_id = window_id or (task.window if task else None)
        target = None
        if window_id:
            target = next((w for w in wins if w["id"] == window_id), None)
        if target is None:
            target = next((w for w in wins if w.get("active")), None)
        answer: dict = {"window": (target or {}).get("title"), "text": []}
        if target:
            if not target.get("active"):
                # OCR reads pixels: an unfocused window can be covered.
                self._call("focus_window", {"window_id": target["id"]})
                target = next((w for w in (self._call("list_windows", {}).get("windows") or [])
                               if w["id"] == target["id"]), target)
            out = self._call("read_window", {"window_id": target["id"], "raise_first": False})
            lines = []
            for line in out.get("lines") or []:
                text = (line.get("x") or "").strip()
                if text and text not in lines:
                    lines.append(text[:160])
            answer["text"] = lines[:10]
            answer["lines_total"] = len(lines)
        if not answer["text"]:
            desc = self._call("describe_screen", {})
            answer["controls"] = [(c.get("t") or "")[:80] for c in (desc.get("controls") or [])
                                  if c.get("t")][:10]
            answer["app"] = (desc.get("active") or {}).get("app")
        return {"ok": True, "answer": answer}


def _detail(out: dict) -> dict:
    """A step's own short readback, capped so the task report stays small."""
    keep = {}
    for key in ("window", "title", "url", "clicked", "text", "count", "quiet",
                "resumed"):
        value = out.get(key)
        if value in (None, [], {}, ""):
            continue
        keep[key] = value if not isinstance(value, str) else value[:120]
    return keep


# -- learned plans -----------------------------------------------------------

def learned_plan(memory, goal: str) -> tuple[str, list[dict]] | None:
    """A previously successful plan for this goal (exact, else close)."""
    want = norm_goal(goal)
    if not want:
        return None
    best: tuple[float, str] | None = None
    for row in memory.skills():
        desc = str(row.get("description") or "")
        if not desc.startswith("task:"):
            continue
        stored = desc[5:].strip()
        if stored == want:
            skill = memory.skill(row["name"])
            return (row["name"], skill["steps"]) if skill else None
        score = jaccard(stored, want)
        if score >= 0.75 and (best is None or score > best[0]):
            best = (score, row["name"])
    if best:
        skill = memory.skill(best[1])
        if skill:
            return best[1], skill["steps"]
    return None


_RUNNER: Runner | None = None


def configure(registry: dict, memory) -> Runner:
    """Bind the process-wide runner to this daemon's registry (one per daemon)."""
    global _RUNNER
    if _RUNNER is None or _RUNNER.registry is not registry:
        _RUNNER = Runner(registry, memory)
    return _RUNNER


def runner() -> Runner:
    if _RUNNER is None:
        raise RuntimeError("tasks.configure(registry, memory) was not called")
    return _RUNNER
