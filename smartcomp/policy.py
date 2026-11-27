"""Tiny learned plan router: a bag-of-words classifier, not a language model.

The executor in `tasks.py` understands a handful of plan shapes (open an app,
go to a URL, click a label, read the result). Deciding which shape a goal
needs is usually a verb match, but agents phrase the same intent differently
("check my profile", "pull up instagram", "show me the page"). This module
learns that mapping from the daemon's own completed tasks.

It is deliberately small: sparse word -> class weights (one score per
template), multinomial softmax, one SGD step per observation, saved as JSON
next to the token. A few hundred words is tens of KB and well under a
millisecond to score -- it cannot understand a sentence, and it does not try
to; it only picks among the templates the executor already implements.

Training data is the task log itself, so the model improves with use and
never blocks a task: with no weights it returns None and the parser decides.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

TEMPLATES = ("open", "goto", "click", "read")
LR = 0.15
MAX_WORDS_PER_CLASS = 4000


def tokens(goal: str) -> list[str]:
    """Lowercase words plus the first word doubled: 'open x' and 'visit x'
    share the shape of an imperative, and that signal is worth keeping."""
    words = [w for w in "".join(c if c.isalnum() else " " for c in goal.lower()).split() if w]
    if not words:
        return []
    return words + [f"verb:{words[0]}"]


class Router:
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.weights: dict[str, dict[str, float]] = {t: {} for t in TEMPLATES}
        self.bias: dict[str, float] = {t: 0.0 for t in TEMPLATES}
        self.examples = 0
        self.correct = 0
        self.updated = 0.0
        self.load()

    # -- model ------------------------------------------------------------
    def _scores(self, toks: list[str]) -> dict[str, float]:
        return {t: self.bias[t] + sum(self.weights[t].get(w, 0.0) for w in toks)
                for t in TEMPLATES}

    def predict(self, goal: str) -> tuple[str, float] | None:
        """(template, probability) or None when there is nothing learned."""
        if self.examples == 0:
            return None
        toks = tokens(goal)
        if not toks:
            return None
        scores = self._scores(toks)
        top = max(scores.values())
        exps = {t: math.exp(s - top) for t, s in scores.items()}
        total = sum(exps.values())
        best = max(exps, key=exps.get)
        return best, exps[best] / total

    def observe(self, goal: str, template: str, ok: bool) -> None:
        """One SGD step. A failed task trains the other templates, which is
        what stops a plausible-looking wrong route from being reinforced."""
        if template not in TEMPLATES:
            return
        toks = tokens(goal)
        if not toks:
            return
        scores = self._scores(toks)
        top = max(scores.values())
        exps = {t: math.exp(s - top) for t, s in scores.items()}
        total = sum(exps.values())
        probs = {t: e / total for t, e in exps.items()}
        if self.predict(goal) is not None:
            self.examples += 1
            if probs[template] == max(probs.values()):
                self.correct += 1
        else:
            self.examples += 1
        for t in TEMPLATES:
            target = 1.0 if (t == template and ok) else 0.0
            grad = (target - probs[t]) * LR
            self.bias[t] += grad
            table = self.weights[t]
            for w in toks:
                table[w] = table.get(w, 0.0) + grad
            if len(table) > MAX_WORDS_PER_CLASS:  # keep the strongest words
                keep = sorted(table.items(), key=lambda kv: -abs(kv[1]))[:MAX_WORDS_PER_CLASS]
                self.weights[t] = dict(keep)
        self.updated = time.time()
        self.save()

    # -- persistence ------------------------------------------------------
    def load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return
        weights = data.get("weights") or {}
        self.weights = {t: {str(k): float(v) for k, v in (weights.get(t) or {}).items()}
                        for t in TEMPLATES}
        self.bias = {t: float((data.get("bias") or {}).get(t, 0.0)) for t in TEMPLATES}
        self.examples = int(data.get("examples", 0))
        self.correct = int(data.get("correct", 0))
        self.updated = float(data.get("updated", 0.0))

    def save(self) -> None:
        if not self.path:
            return
        payload = {"version": 1, "templates": list(TEMPLATES), "weights": self.weights,
                   "bias": self.bias, "examples": self.examples,
                   "correct": self.correct, "updated": self.updated}
        tmp = self.path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(payload, separators=(",", ":")))
            os.replace(tmp, self.path)
        except OSError:
            pass

    def status(self) -> dict:
        size = 0
        if self.path and self.path.exists():
            size = self.path.stat().st_size
        return {
            "examples": self.examples,
            "accuracy": round(self.correct / self.examples, 3) if self.examples else None,
            "words": {t: len(self.weights[t]) for t in TEMPLATES},
            "bytes": size,
            "updated": self.updated or None,
            "templates": list(TEMPLATES),
        }
