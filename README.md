# SmartComp — local perception tier

A local daemon that gives an AI agent **screen perception and input**, in a JSON
form small enough to sit in a prompt. Perception is the M1 slice of the design
plan: capture, tile-level change detection, OCR, an icon/button detector, and a
`get_changes` tool served over MCP on localhost. On top of it sits an action
layer (XTEST pointer/keyboard, clipboard, window management, allowlisted app
launch) and a small memory tier (skills, observations, per-tool stats).

## Availability

Free forever — fully public and free, built on the open tools available on the
machine. **Now supported on Linux/X11** (this target: Debian-family, XFCE, X11 —
Debian-family packages and systemd user units). **Windows and
macOS support launches in the next update**, now with additional features (see
the release notes / roadmap for details). Isolated agent sessions need `Xvfb`
(plus an X11 WM such as `xfwm4`) installed.

Built and measured on the target box: Kali, X11/XFCE, i3-7100U (2 cores / 4
threads), 7.7 GB RAM, 1920x1080.

## Video / voice understanding (low RAM)

`media_understand(source)` turns a YouTube/video URL or local file into small
structured JSON for the LLM — **no images** — using captions first (zero RAM),
Whisper tiny int8 on 16 kHz audio in 30 s chunks (~40 MB weights, ~300 MB RAM;
the unchunked CTranslate2 mel buffer is the documented OOM trap), and ffmpeg
frame sampling + Tesseract OCR for burned-in text (reels, songs, captions).
Returns metadata (title/duration/uploader), timestamped transcript, frame
text, hashtags and URLs. `media_capabilities()` reports the engines. One model
is loaded at a time, frames are released per loop, and temp files are cleaned.

Verified live: a Firefox YouTube page read by smartcomp + `media_understand`
on the same URL returned title/duration/uploader and 52 timestamped caption
lines in ~21 s; a spoken audio clip transcribed correctly through Whisper
tiny (`mode: whisper-tiny`, 4.7 s); two more YouTube videos returned 60
caption lines each. **Honest limits**: this box's yt-dlp (no JS runtime)
gets HTTP 403 downloading YouTube *video/audio data*, so frame sampling and
the Whisper fallback work on local/direct media but not on YouTube videos —
captions remain the path there; Instagram reels are behind a login wall, so
`media_understand` cannot fetch them (an authenticated session is needed).

## Multiple isolated sessions (virtual cursor per agent)

One X11 display has exactly one pointer, so agents that share a desktop fight
over the cursor and see each other's windows. `session_create(name)` gives
each agent a **private Xvfb desktop** (1920x1080) with its own window manager,
its own XTEST cursor, its own daemon/port/token, and its own state dir. XTEST
is display-local, so a session's clicks never touch the host pointer or other
sessions. The virtual display has no screen lock — **sessions keep working
while the host is locked, blanked or asleep, without waking or unlocking the
host**. Connect a second LLM with `session_endpoint(name)`.

`session_align()` merges every session's skills/observations into a shared
pool and pushes the pool back to all sessions; `session_train()` distils each
session's successful actions into skills and feeds the pool — the fleet gets
better the more it is used. `session_list()` / `session_close()` manage the
lifecycle. Verified live: two independent cursors (host vs `:1000`), a Mousepad
launched on the private display and perceived there (28 elements), healthy
daemon on the session port, alignment/training round-trip, clean teardown.

### Visible sessions (watchable, still isolated)

A headless session is invisible to the human. `session_create(name,
visible=true)` starts the session's display as an **Xephyr window on the host
screen** instead: still its own X server, its own XTEST cursor and its own
apps, but two agents can be watched working side by side (windows are placed
in screen slots with a ConfigureWindow request — no synthetic input). Input
stays inside the nested server, so nothing a session does can reach the host
apps.

That nesting is also the *only* delivery for a second cursor. Multi-pointer X
(XInput2 master pointers) genuinely works — a second real cursor, its own
clicks and keys, core pointer unmoved — but creating a master on a **live**
display crashes Qt6 apps (`libQt6XcbQpa`, observed as three qterminal
segfaults in one session). `smartcomp/mpx.py` holds the working wrapper
(`available`, `ensure`, `list_masters`, `destroy`, with warp/button/key on the
private connection) for a throwaway display; nothing imports it, and it must
not be pointed at the user's own screen.

`launch_app("firefox", ["https://example.com"], profile="real")` clones the
user's real profile (cookies, logins, extensions; cache dirs skipped) into the
session, so the nested browser is already signed in and the running browser is
left alone. `draw_stroke(element_id, shape="circle"|"zigzag"|"spiral"|"line")`
drags a path computed inside an element's box — canvas and webapp drawing with
the agent naming an element rather than emitting pixels.

Verified live: two visible sessions (Firefox and Brave, both on cloned
profiles) side by side on `sketchpad.app`, `draw_stroke` returning `ok:true`
with element deltas, then both sessions closed with no Xephyr process or
browser left behind.

## What's new in this update (v2 — dual vision + popup awareness)

The first public release addressed elements, not pixels. This update adds
**image understanding** and **popup awareness** while keeping the token cost
low and the contract unchanged (the agent still never sees pixels).

* **Two readers, one merged report** — `explain_screen()` runs smartcomp's
  OCR + GPA detector and **Mirage** (Cactus Compute's 22 MB, 51M-parameter
  on-device vision model, `cactus-compute`) on the **same raised window**, and
  the decider merges both answers (text, scene, objects, colors, buttons) into
  one first-class JSON block. Measured: on dense terminal text OCR wins; on
  browser pages Mirage reads the prose OCR misses; on the same screen each
  reader sees things the other doesn't, so merging is strictly more complete.
  Mirage is optional (install with `pip install cactus-compute`); the daemon
  degrades to OCR-only when it is absent.
* **`popups()` / `watch_popups()`** — a first-class popup block instead of
  burying dialogs in a delta. Detects (1) small dialog windows, (2)
  override-redirect X windows the WM client list omits (notifications/menus),
  (3) in-page warning banners the LLM would otherwise miss (Firefox's "files
  in use by another application", Brave's "didn't shut down correctly"), and
  (4) button-like elements **inside** the popup's rect. `watch_popups()`
  blocks until one appears. Both readers describe a found popup and the
  decider merges the result; buttons come back as element ids to act on.
* **New-window awareness** — a window appearing after an action (e.g. a
  Google sign-in popup with an unremarkable title/size) is now reported in
  the action's `changed.windows` block and by `popups()` as
  `source: "new-window"` with id/app/title/box.
* **Video/voice understanding** — see the dedicated section below.
* **No coordinates, even for context menus** — `context_menu()` now takes an
  element id/label or picks a blank spot itself; `x`/`y` are gone from its
  schema. `click_element()` snaps tiny OCR glyphs (a 5x9 "x" close button) to
  the overlapping detector icon that is the real hit target — verified by
  opening and closing a browser tab by element id.
* **Self-improvement loop** — `session_end(auto_improve=True)` (default)
  distils each session's successful actions into a replayable skill, which
  gains use/win counts on replay (`run_skill`). Combined with `stats()`,
  `remember`/`recall` and session traces, every use leaves the daemon with a
  reusable flow — it gets better the more it is used.
* **OCR robustness** — the confidence rescue now runs on scan paths at any
  window size (maximized pages used to get none); 2x-upscale and threshold
  variants were measured worse and deliberately not shipped.

Requires `cactus-compute` for the Mirage half of `explain_screen`/`popups`;
everything else works with the base install. Install with
`pip install cactus-compute` — weights (~22 MB) are cached on first use.

```
screen ──mss──► native frame ──resize──► 960x540 logical frame
                     │                          │
                     │                    144-tile diff ──► changed regions
                     │                          │
                     └──► OCR on native crops ──┴──► detector ──► element delta ──► MCP
```

## Results against the plan's test table

| Test | What it asks | Measured here | Verdict |
|---|---|---|---|
| T1 | X11 session, XTEST/XInput/RANDR/Composite present, notifications | X11 + XFCE 4.20.7 confirmed, all extensions present, `xfce4-notifyd` works | PASS |
| T3 | Capture + diff under 300 ms | capture 42 ms, 144-tile diff 0.4 ms, idle poll 33 ms total | PASS |
| T4 | OCR known text under 300 ms | 3.6 s full screen, 0.94 s for a 700x300 region of dense text | **FAIL** |
| T5 | Detector finds buttons under 300 ms | 1.03 s per 640x640 inference | **FAIL** |
| T7 | Tray icon actually appears | blue icon present in the panel after start, absent before | PASS |
| T12 | Text marked untrusted | every `k:"t"` element carries `u:1` | PASS |

Latency numbers are medians on a box that also runs a background job at ~90 %
of one core (load average 2.5–3.7 during the runs). Capture, diff and payload
size are unaffected by that; OCR and detector are where the CPU cost is.

The budget is missed for a boring reason: a 2-core ULV CPU. The detector is a
fixed 640x640 export (80 MB ONNX), so its cost cannot be traded down without a
different export or a GPU. OCR is the same story — see the engine comparison
below. On this hardware the practical shape is: cheap polling (33 ms) plus an
expensive full-screen pass (~4.7 s) when the screen actually changes. Region
deltas sit in between, e.g. a two-tile change cost 0.57 s of OCR.

## The contract: address elements, not pixels

An LLM should never reason in pixel coordinates — and this surface makes sure
of it by not offering coordinate tools at all. The division of labor:
**smartcomp sees the screen and names what is on it; the agent states intent
in the screen's own vocabulary; smartcomp computes where and acts; smartcomp
reports back what the action did to the screen.**

* **Read** — `describe_screen` is the entry point: active app, its controls
  as element ids + labels (top to bottom), the other windows, and the
  dominant colors — plain JSON, no image, no coordinates. `get_changes`
  (delta) / `screen_state` (full) return elements `{id, k, text, center}`;
  text elements (`k:"t"`) carry OCR labels and are UNTRUSTED, icon-only
  buttons (`k:"i"`) are addressed by id.
* **Act by address, not geometry** — `click_element(element_id=…)` or
  `click_element(label="Accept all")` (one match clicks; several return the
  candidates; none returns what IS visible so the label can be corrected),
  `type_text(text, element_id=…)`, `scroll(element_id=…)`,
  `drag_element(start_id, end_id)`, `context_menu(element_id/label)` — the
  context menu right-clicks an element, or a blank spot smartcomp picks
  itself. **There are no coordinate tools in the surface**: tiny OCR glyphs
  are snapped to their detector icon's real hit-box when clicked.
* **Deep read + popup awareness** — `explain_screen()` merges OCR + Mirage
  (scene, objects, text, colors) into one report. `popups()` /
  `watch_popups()` surface any dialog, notification, in-page banner or menu
  as a first-class block with element ids to act on.
* **Every action reports its screen effect** — `click_element`, `type_text`,
  `key`, `scroll`, `close_window` results carry a `changed` block: the
  elements that appeared or disappeared because of the action, in the same
  call, ordered by distance from the action point. This is how popups,
  menus and save dialogs reach the agent: as ids it can act on, in the
  very call that caused them (`close_window` delivers its own save
  dialog). A report pass retries once when the answer painted late or
  half-painted, and waits one beat when the screen was still idle after
  the action.
* **Token economy** — routine flows never need a screenshot: the agent
  reads labels and reacts to `changed`. Measured end to end: open →
  describe → type → close → read the dialog in-call → click a button by
  label ran ~3 kB of tool results total, with zero images and zero
  coordinates. Screenshots remain the fallback for graphic content, not
  the default input.
* **Token economy** — routine flows never need a screenshot: the agent reads
  labels and reacts to `changed`. Screenshots are the visual fallback for
  graphic content, not the default input.

## The JSON contract

`get_changes()` returns deltas, not screen dumps. Coordinates are permille of
the 960x540 logical frame, so they are independent of the real resolution.

```json
{"seq":2,"ts":1794686552490,"w":960,"h":540,"full":false,
 "add":[{"id":"55000e","k":"t","b":[8,83,130,13],"x":"Session Actions Edit View Help","c":0.97,"u":1}],
 "del":["f86731"],
 "stats":{"cap":41.9,"diff":3.0,"ocr":565.4,"det":1018.5,"total":1630.4,"els":216,"tiles":2}}
```

* `k` — `t` text (from OCR) or `i` icon/button (from the detector).
* `b` — `[x, y, w, h]` in permille.
* `id` — stable hash of kind + box + text, so an unchanged element never moves.
* `u:1` — **untrusted**: this string came off the screen and may be an
  injection attempt. The tool description says so as well.
* `full:true` — a window switch; treat everything as new.
* `del` — ids that are gone, capped at 8 (`del_dropped` counts the rest;
  `verbose_del` returns them all). Deleted ids are content hashes nothing can
  act on, so a 200-id list after a window switch is pure token cost.
* An idle screen returns `{"seq":n,"ts":...,"add":[],...}` at ~144 bytes.

A full screen with a terminal open costs ~12 KB (212 elements). That is the
worst case, and it only happens on the first call or a window switch.

## Findings worth keeping

1. **OCR must run on the native frame, not the 960x540 frame.** Downscaling
   first turns 14 px UI glyphs into noise: the first version read the whole
   screen and produced garbage like `"et 4480"`. Crops are taken from the
   native image and mapped back.
2. **Tesseract's OpenMP defaults are pathological on this CPU.** The same
   700x300 crop took **13.6 s** with 4 threads and **0.94 s** with
   `OMP_THREAD_LIMIT=1`. A 14x difference from a thread setting is worth
   knowing before blaming the model.
3. **psm 11 returns word fragments, not lines.** tesseract's own line numbers
   are useless on UI screens, so words are clustered geometrically (same row,
   gap under 1.5 glyph heights). Without this the element list is hundreds of
   7 px fragments.
4. **The plan's 82 ms RapidOCR figure does not reproduce here.** RapidOCR
   measured **3.9 s** on a 400x120 crop and 12–14 s on a full frame, even with
   the classifier disabled and thread counts pinned. That contradicts the
   design document's T4 assumption, so tesseract is the default engine and
   `SMARTCOMP_OCR=rapidocr` opts into PP-OCR. The wheel also would not download
   through pip (the CDN drops the connection); `scripts/fetch_rapidocr.sh`
   resumes it instead of failing.
5. **Detector threading: 3 threads beat 1 and 4.** 640x640 inference measured
   1.5 s (1 thread), 1.07 s (2), 0.98 s (3), 1.53 s (4) — oversubscribing a
   2-core box is slower than using fewer threads.
6. **The tray has to be XEmbed here.** The panel owns `_NET_SYSTEM_TRAY_S0`, so
   `Gtk.StatusIcon` shows pixels; Xfce 4.20 ships no SNI panel plugin. A
   StatusNotifier watcher does exist on the session bus, so the earlier
   assumption that "SNI is unavailable" was only half right — no *panel host*
   was registered, which is what matters.
7. **A clock tick must not cost a scan.** Regions below `MIN_REGION_PX`
   (4000 logical px²) skip OCR and the detector entirely, which is the
   difference between a 1.3 s idle poll and a 33 ms one.
8. **Port 8765 is taken** by an unrelated local daemon on this machine, so
   SmartComp listens on **8791** (`SMARTCOMP_PORT` overrides).
9. **`scroll` was guaranteed-broken by one argument-binding bug.**
   `inp.scroll(amount, axis=axis, *px)` unpacks the coordinates positionally
   into `axis`'s slot, so every call died with
   `TypeError: Input.scroll() got multiple values for argument 'axis'`
   before any X event was emitted (0/4 in the action log, ~24 ms each). The
   fix binds the coordinates by keyword. `amount` is also capped at 20 wheel
   ticks per call now (`clamped: true` comes back when the cap bit, and the
   model can call again) because `scroll(amount=400)` otherwise means 400
   press/release pairs.
10. **`clipboard_set` was 0/9 because of pipe inheritance.** xclip forks and
    keeps owning the selection; with `subprocess.run(..., capture_output=True)`
    the forked owner inherits our stdout/stderr pipes, EOF never arrives, and
    every call eats the full 5 s timeout. Popen with `stdin=PIPE`, DEVNULL for
    the rest and a wait for the short-lived parent: a set→get round trip now
    measures ~130 ms. Design note: a selection-serving child must never share
    pipes with the RPC.
11. **Failure reasons are first-class data now.** The action log records the
    error text (exception class + message, or the `error` field of an
    `ok:false` result, capped at 200 chars) and `stats()` histograms it —
    "screenshot wins 60/558" becomes "screenshot fails with XError when the
    locker grabs the display".
12. **Window-scoped screenshots died on X11 BadMatch (major opcode 73).**
    The grab rectangle went to XGetImage unclamped; a window whose
    translated origin is negative or runs past the root edge (frame
    offsets, a window dragged partly offscreen) failed every time. The
    rectangle is clamped to the virtual screen now, an entirely-offscreen
    window is reported instead of crashing, and a grab that still fails
    falls back to a full-screen grab cropped in software.
13. **A near-empty `read_window` result is flagged, not served as truth.**
    Dark/graphic UIs (browsers) can defeat tesseract completely — a real
    Firefox read came back as the single line `ao x`. Such reads carry
    `low_yield: true` plus a hint pointing at screenshot / keyboard-driven
    reading / get_changes. The OCR ceiling itself stands; the real fix for
    browser pages is a DOM/CDP companion (still not built).
14. **Stats tell lifetime from current.** `stats()` reports a `recent`
    window (24 h by default, `hours` parameter) next to the lifetime
    aggregates: screenshot's lifetime 11 % over 561 calls otherwise reads
    as still-broken while the fixed path scores 100 %.
15. **Sessions rotate on task gaps.** A gap longer than
    `SMARTCOMP_SESSION_GAP_MINUTES` (default 30) between two actions ends
    the session and starts a fresh one, so a daemon that lives for days
    does not blend unrelated tasks into a single trace.
16. **launch_app resumes instead of duplicating.** An app with a live
    window is focused, not started again — two calls used to mean two
    Firefoxes. A running-but-windowless app is reported instead of
    stacked, and `force=true` is the explicit way to start a second
    instance.
17. **App and tab census (`app_status`).** One call reports, per
    allowlisted app, the live process count (matched by /proc executable
    identity, never by title), the mapped windows with the focused one
    marked — and for Firefox the exact per-window tab counts read out of
    its own session store (`recovery.jsonlz4`, ~15 s save interval), so
    "how many tabs did I open?" is answerable without an extension or CDP.
    Tab counts are gated on a live process: a fresh `recovery.jsonlz4`
    outlives a crash or a close, and it would otherwise describe tabs
    nothing can see.
18. **Launched apps died with every daemon restart.** The daemon is a
    systemd user service and systemd stops by cgroup: an app spawned as a
    child (even `start_new_session`, which detaches the session, not the
    cgroup) was swept away by the next restart's stop kill — measured live,
    a Firefox died mid-session with no OOM and no crash of its own, which
    is also what made "resume the open app" look unreliable. Apps now run
    as their own transient user service (`smartcomp-app-*`) via
    `systemd-run --user`, outside the daemon's cgroup, with the display
    environment forwarded and a fallback to the old detached child when no
    user manager exists. Their output goes to `journalctl --user -u
    <unit>` instead of /dev/null.
19. **Actions now carry the screen's answer, and the agent addresses
    elements, not pixels.** Input actions used to return bare results, so
    every "click this" needed a follow-up `get_changes` call to learn what
    happened, popups appearing between actions were invisible until
    polled, and typing into a field meant coordinate arithmetic. Now
    every input action attaches a `changed` block (elements that appeared
    or vanished, capped like the delta), `click_element` takes a label as
    well as an id (ambiguous labels are refused with the candidates, not
    guessed), `type_text`/`scroll`/`drag_element` take element addresses,
    and idle perception passes skip the detector (detections only feed
    changed rects, so an unchanged screen cannot gain icons -- the
    detector was a fixed tax on every poll, measured at ~1 s of the
    ~1.6 s idle pass).
20. **Popup reports have a measured limit: a popup opening amid a large
    repaint can be missed.** A click that both scrolled a page AND opened
    its context menu merged the menu into the one giant changed region,
    and whole-region OCR lost the small menu text (measured live on a
    Firefox page: menu open on screen, zero of its items in the element
    store, even after the report's retry -- the retry catches a popup
    that painted *after* the capture, not text OCR failed on). When the
    popup's region changes in isolation it works end to end (desktop
    right-click menu: reported, addressed by label, item clicked). The
    planned fix is action-point priority OCR: a small deep pass centred
    on where the action happened, regardless of what else repainted.
21. **`key` now speaks the agent's vocabulary.** X11 keysym lookup is
    case-sensitive and canonical, so the natural names agents write
    ("backspace", "esc", "enter", "pgdn") failed with "unknown key" --
    26 backspaces to clear a field did literally nothing while reporting
    ok-looking errors. Alias table plus a case-insensitive keysym sweep;
    measured: the same 26 presses clear the field.
22. **The coordinate tools are gone from the surface, on purpose.**
    `mouse_move`, `click`, `double_click`, `mouse_down`, `mouse_up` and
    free-form `drag` were removed from the MCP registry -- an agent that
    can see pixels will use pixels, so the surface must not offer them.
    Element addressing (`click_element`, `type_text(element_id=…)`,
    `scroll(element_id=…)`, `drag_element`) is the only road; the single
    deliberate coordinate action is `context_menu(x, y)` for blank space
    that has no element. Surface went 43 -> 39 tools.
23. **`describe_screen` replaces the screenshot as the default read.** One
    call returns the active app + title, the controls inside it (ids +
    labels, top to bottom, capped at 12 with `controls_dropped`), the
    other windows, and the dominant colors (quantized BGR -> hex with
    pixel share, so the agent can say "the red button" without seeing
    one). Measured: 1.3 kB for a full mousepad description. The planned
    small classifier model does not exist upstream (the fetch script
    silently saved a 404 body as `mobilenetv3_small.onnx` -- the release
    only ever shipped the detector), so the describer is built from real
    signals only: OCR + GPA detector + colors + window identity. The
    fetch script now refuses to save a 404 (HEAD check, `-f`, and a
    1 MB sanity floor) and the fake model is deleted.
24. **Action reports now survive both late-paint traps.** A close's save
    prompt maps after an IPC round trip: the report pass ran 55 ms after
    the close on a fully idle screen (no tiles moved yet) and the prompt
    painted after it -- invisible in the very call that opened it. A
    report with an action point now waits one beat and re-passes when the
    screen is stone idle, and retries once when elements appeared but
    none landed near the action point (measured live: Firefox repaint
    noise reported while the save dialog was one beat late). End state
    measured live: `close_window` delivers its own save dialog
    ("Don't Save" as an id) in the same call; the whole
    open->describe->type->close->dialog->dismiss cycle ran ~3 kB of tool
    results with zero screenshots and zero coordinates.
25. **Dense-text OCR: measured the ceiling, changed for the gap.** I built
    a synthetic dense page (two columns, 8 px text, dates, snippets) and
    scored smartcomp's OCR against ground truth (mean CER over 84 truth
    lines): **current pipeline 0.514**, PSM 6 0.525, PSM 3 (85 lines),
    plain Otsu 0.518, adaptiveThreshold 0.543, 2x upscale + Otsu 0.767
    (2 lines lost — much worse). MedianBlur destroyed the small strokes
    (0.287 CER but 10 lines kept — discarded). Real-terminal
    `read_window` output stays readable with the classic small-text noise
    (H for t, | for I, "Ka le" for Kaggle) — that floor is Tesseract's on
    ~9 px text. Change shipped: the strip rescue in `ocr._read_tesseract`
    now runs on scan paths at any window size when the main pass has
    failures (`deep` no longer gates it); measured neutral on the synthetic
    and the unit-tested reason it now matters: on maximized Firefox pages
    (over the 1.5 M px alt budget) scan passes used to get zero rescue
    after a failed main-pass row.

## Licensing

* Detector weights: **Salesforce GPA-GUI-Detector, MIT**, exported to ONNX
  (chosen over OmniParser V2, whose weights are AGPL-2.0 and unusable for
  shipping without a licence decision).
* OCR: tesseract (Apache-2.0) by default, RapidOCR (Apache-2.0) optional.
* The ONNX file used here comes from a community export, not from Salesforce
  directly. **Before distributing the weights, re-export from the original MIT
  release in this repo and keep the licence text with it.**

## Privacy and security

* Binds `127.0.0.1` only, plus a bearer token generated on first run into
  `~/.local/state/smartcomp/token` (mode 0600). Unauthenticated requests get
  `401` — covered by `tests/mcp_smoke.py`.
* Storage is session-scoped and inspectable: screenshots go under
  `~/.local/state/smartcomp/sessions/<id>/` and are deleted at `session_end`
  (pruned after 24 h if a session was never closed); the local SQLite DB
  keeps the per-tool action log — its args include any text typed via
  `type_text` — for stats and traces. The element store itself lives in
  memory and holds only what is currently on screen.
* Screen text is data, never instructions: it is flagged `u:1` and never
  executed, matched against a shell, or passed to a tool.

## Running it

```bash
cd /home/damon/smartcomp
.venv/bin/python -m smartcomp.daemon serve      # MCP on 127.0.0.1:8791
.venv/bin/python -m smartcomp.tray              # panel icon
.venv/bin/python -m smartcomp.tray --diagnose   # which tray host exists
.venv/bin/python -m smartcomp.daemon once       # one JSON pass
.venv/bin/python -m smartcomp.daemon bench      # T3/T4/T5 measurement
.venv/bin/python tests/mcp_smoke.py             # end-to-end MCP client
.venv/bin/python -m unittest discover -s tests  # unit tests
```

## Install on this machine

One idempotent script does everything — venv, editable package, user service
(enabled at login), the `smartcomp` CLI on PATH and the agent endpoint file:

```bash
/home/damon/smartcomp/scripts/install.sh
```

Afterwards:

* `smartcomp serve|once|bench|state|tray` works from any directory
  (`~/.local/bin/smartcomp`).
* The daemon runs as the user service `smartcomp.service` (restart:
  `systemctl --user restart smartcomp`, log: `/tmp/smartcomp_server.log`).
* `~/.local/state/smartcomp/mcp.json` (mode 600) holds everything an agent
  needs: URL, ready-to-use Authorization header, token path, docs.

## Connect any MCP agent

The server speaks streamable HTTP with a bearer token. Any MCP client that
supports streamable HTTP can attach with:

```json
{
  "mcpServers": {
    "smartcomp": {
      "type": "streamable-http",
      "url": "http://127.0.0.1:8791/mcp",
      "headers": {
        "Authorization": "Bearer <contents of ~/.local/state/smartcomp/token>"
      }
    }
  }
}
```

Clients that cannot read files can parse `~/.local/state/smartcomp/mcp.json`
directly — it already contains the exact `url` and `headers` to use. Verify a
connection from anywhere:

```bash
curl -s http://127.0.0.1:8791/mcp \
  -H "Authorization: Bearer $(cat ~/.local/state/smartcomp/token)"
```

The venv is created with `--system-site-packages` because numpy, onnxruntime,
mss, OpenCV, httpx and `mcp` are already installed for the system python; only
`pystray` is added.

## Unattended operation (locked or unlocked)

The daemon is built to keep an agent's work going with no human at the machine
— locked screen or not. It rests on three mechanisms, all verified live on this
box:

| Piece | What it does |
|---|---|
| `keep_awake` inhibit | DBus `ScreenSaver.Inhibit` stops the *idle* lock, and `xset` turns off DPMS blanking + the X screensaver timeout for the daemon's lifetime. Enabled automatically at daemon start; `SMARTCOMP_AWAKE=0` opts out. Crash-safe: the inhibit is owned by the daemon's bus connection, so it dies with the process. |
| Lid policy (`apply_lock_policy`) | xfce4-power-manager's stock lid trigger is `LID_TRIGGER_LOCK_SCREEN` (enum value 3, per `xfpm-enum-glib.h`) — and per the Xfce docs a Lock call "bypasses all Inhibit requests", which is exactly how a closed lid locked this box mid-run. The daemon moves both `lid-action-on-ac/on-battery` to `4` (NOTHING), records the previous values in `~/.local/state/smartcomp/lid_policy.json`, and restores them when unattended mode is disabled. The tool side is consent-gated: `lid_lock(on=false)` without `confirm=true` is refused (it removes the last lock on lid close), and `session_end` restores the stock policy by default. |
| Lock watchdog | A daemon thread checks every ~15 s: if a lock appears anyway (manual Lock, a lid event, a second locker) it releases it via `lock.hard_unlock()` and re-asserts the xset state (xfpm re-enables DPMS on lid events). Recovery is logged to the service log. |
| `unlock_screen` | On-demand recovery: asks the screensaver to deactivate over its own documented DBus API — verified live: xfce4-screensaver hands the screen back with **no password and nothing typed**. If a locker refuses deactivation, it falls back to typing a user-stored credential (below). |
| `wake` / `lock_state` | Force a DPMS-blanked display back on (never unlocks) / report whether a lock currently holds the screen (also in `health`). |

**Why recovery needs no password here.** The lock on this desktop is held by
xfce4-screensaver, which implements `SetActive(FALSE)` as "deactivate" — the
same call `xfce4-screensaver-command --deactivate` makes. Live test
(2026-11-16): screen locked via `SetActive(true)` → watchdog released it in
≤10 s via `SetActive(false)` on `org.xfce.ScreenSaver /org/xfce/ScreenSaver`,
desktop and all windows intact. The credential path exists for lock
implementations that refuse deactivation (light-locker, PAM-verified states).

**Optional stored credential.** Only needed as a fallback; created by you, the
password never crosses the tool API and is never logged:

```bash
install -m 600 /dev/null ~/.local/state/smartcomp/unlock
# then edit that file to hold your login password (single line)
```

A file that is group- or world-readable is refused outright rather than used.

**Security trade-off, stated plainly.** While the daemon runs unattended, the
desktop does not stay locked: idle locking is inhibited, the lid no longer
locks, and any lock that does appear is released within seconds. That is the
point of the feature — an agent that must keep working with no human present —
but it means the machine effectively runs unlocked. Stop the daemon
(`systemctl --user stop smartcomp.service`) or `keep_awake(false)` to return to
normal behaviour: the inhibit releases, the lid policy is restored, and the
screensaver behaves stock again.

## Layout

| File | Role |
|---|---|
| `smartcomp/capture.py` | mss capture, downscale, 144-tile diff |
| `smartcomp/ocr.py` | tesseract / RapidOCR engines, word→line clustering |
| `smartcomp/detector.py` | GPA ONNX decode (letterbox, NMS, transpose-safe) |
| `smartcomp/perception.py` | region clustering, element ids, delta JSON |
| `smartcomp/apps.py` | app census, launch resume semantics, Firefox tab counts |
| `smartcomp/awake.py` | screensaver inhibit + DPMS control for unattended runs |
| `smartcomp/lock.py` | lock-state detection + credential-based unlock recovery |
| `smartcomp/sessions.py` | private Xvfb/Xephyr desktop per agent, daemon + token per session |
| `smartcomp/mpx.py` | multi-pointer X wrapper (own cursor per master); unused, see above |
| `smartcomp/server.py` | FastMCP streamable HTTP + bearer middleware |
| `smartcomp/tray.py` | Gtk.StatusIcon tray, host diagnostics |
| `smartcomp/daemon.py` | CLI: `serve`, `tray`, `once`, `bench`, `state` |
| `scripts/fetch_*.sh` | resumable model/wheel downloads for slow links |

`models/` holds `gpa_gui_detector.onnx` (80 MB, single-class icon/button
 detector) and `mobilenetv3_small.onnx` (15-class UI element classifier:
 button / link / text input / ..., downloaded but not wired in yet — it is the
 natural next step for naming `k:"i"` elements).

## Not built yet

The later milestones are deliberately absent: no DOM/browser-grade page
reading (OCR of a downscaled render is the only way to read a page today — a
CDP-based companion tool for Firefox/Chromium is the biggest known win), no
popup policy engine, and no automatic choice between a headless and a visible
session — `visible` stays an explicit per-session decision. The perception
tier came first; the action and memory layers followed it.
