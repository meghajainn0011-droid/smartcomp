"""Unit tests for the parts that do not need a display.

Run with: .venv/bin/python -m unittest discover -s tests -p 'test_*.py'
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import os
import sys
import tempfile
import threading
import time
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smartcomp import apps as apps_mod  # noqa: E402
from smartcomp import config  # noqa: E402
from smartcomp import input as input_mod  # noqa: E402
from smartcomp import tools as tools_mod  # noqa: E402
from smartcomp import popups as popups_mod  # noqa: E402
from smartcomp import sessions as sessions_mod  # noqa: E402
from smartcomp import vision as vision_mod  # noqa: E402
from smartcomp import skillbook as skillbook_mod  # noqa: E402
from smartcomp import tasks as tasks_mod  # noqa: E402
from smartcomp import policy as policy_mod  # noqa: E402
from Xlib import XK  # noqa: E402  (keysym ground truth for alias tests)
from smartcomp import display as display_mod  # noqa: E402
from smartcomp import input as input_mod  # noqa: E402
from smartcomp import memory as memory_mod  # noqa: E402
from smartcomp.display import _parse_xset  # noqa: E402
from smartcomp import lock as lock_mod  # noqa: E402
from smartcomp.capture import Capturer  # noqa: E402
from smartcomp.detector import Detector  # noqa: E402
from smartcomp.ocr import Ocr, TextLine, _cluster_lines  # noqa: E402
from smartcomp.perception import Perception, _eid, _permille  # noqa: E402
from smartcomp.tools import _clip_box, _looks_degenerate  # noqa: E402


def _capturer() -> Capturer:
    """A Capturer with no X connection, for tile-math tests."""
    c = Capturer.__new__(Capturer)
    c._prev_gray = None
    c._cols, c._rows = config.GRID_COLS, config.GRID_ROWS
    c._tw = config.LOGICAL_W // c._cols
    c._th = config.LOGICAL_H // c._rows
    return c


class TestTileDiff(unittest.TestCase):
    def test_first_frame_marks_everything_changed(self):
        c = _capturer()
        gray = np.zeros((config.LOGICAL_H, config.LOGICAL_W), np.uint8)
        self.assertEqual(int(c.tile_matrix(gray).sum()), config.GRID_COLS * config.GRID_ROWS)

    def test_identical_frame_marks_nothing(self):
        c = _capturer()
        gray = np.zeros((config.LOGICAL_H, config.LOGICAL_W), np.uint8)
        c.tile_matrix(gray)
        self.assertEqual(int(c.tile_matrix(gray).sum()), 0)

    def test_single_tile_change_is_localized(self):
        c = _capturer()
        base = np.zeros((config.LOGICAL_H, config.LOGICAL_W), np.uint8)
        c.tile_matrix(base)
        moved = base.copy()
        # Paint the whole of tile (row 2, col 3).
        x, y, w, h = c.tile_rect(2 * config.GRID_COLS + 3)
        moved[y : y + h, x : x + w] = 255
        changed = c.tile_matrix(moved)
        idx = list(np.flatnonzero(changed.reshape(-1)))
        self.assertEqual(idx, [2 * config.GRID_COLS + 3])

    def test_noise_below_threshold_is_ignored(self):
        c = _capturer()
        base = np.zeros((config.LOGICAL_H, config.LOGICAL_W), np.uint8)
        c.tile_matrix(base)
        # A handful of pixels moving is a cursor blink, not a screen change.
        speck = base.copy()
        speck[10:14, 10:14] = 255
        self.assertEqual(int(c.tile_matrix(speck).sum()), 0)


class TestLineClustering(unittest.TestCase):
    def test_words_on_one_row_become_one_line(self):
        words = [
            (10, 20, 30, 12, 90.0, "Hello"),
            (44, 21, 40, 12, 80.0, "world"),
            (10, 60, 25, 12, 70.0, "second"),
        ]
        lines = _cluster_lines(words, 1.0)
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0].text, "Hello world")
        self.assertEqual(lines[1].text, "second")
        self.assertEqual((lines[0].x, lines[0].y), (10, 20))
        self.assertEqual(lines[0].w, 74)

    def test_left_word_merges_into_a_line_seen_first(self):
        # Regression: tesseract reported "control test 42" before "1 smartcomp",
        # and the old gap check only looked rightward, so the line stayed split.
        words = [
            (389, 373, 49, 10, 96.0, "control test"),
            (340, 374, 44, 12, 90.0, "1 smartcomp"),
        ]
        lines = _cluster_lines(words, 1.0)
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0].text, "1 smartcomp control test")
        self.assertEqual(lines[0].x, 340)

    def test_word_joins_its_own_row_not_a_nearby_stray_row(self):
        # Real mousepad read: a garbage row sat above the typed text, and the
        # order-dependent matcher pulled "test 42" onto it, splitting the line.
        words = [
            (13, 39, 44, 15, 40.1, "cy"),
            (74, 39, 46, 15, 44.0, "BR"),
            (13, 71, 7, 10, 70.6, "1"),
            (34, 71, 71, 13, 75.9, "smartcomp"),
            (114, 70, 54, 11, 96.7, "control"),
            (178, 71, 30, 10, 92.9, "test"),
            (218, 68, 16, 18, 91.9, "42|"),
        ]
        texts = [line.text for line in _cluster_lines(words, 1.0)]
        self.assertIn("1 smartcomp control test 42|", texts)

    def test_far_apart_words_stay_separate(self):
        words = [(10, 20, 30, 12, 90.0, "left"), (400, 20, 30, 12, 90.0, "right")]
        self.assertEqual(len(_cluster_lines(words, 1.0)), 2)

    def test_stray_low_confidence_glyph_is_dropped(self):
        words = [(10, 20, 4, 8, 40.0, "|")]
        self.assertEqual(_cluster_lines(words, 1.0), [])

    def test_zero_conf_prefix_words_do_not_kill_the_line(self):
        # Real tesseract psm-11 output (mousepad, dark theme): the row's first
        # words score conf 0.0 while the rest score 96+. The old mean let two
        # broken words drop the whole line below the confidence gate, so a
        # read_window after typing returned only the menu bar and hid real
        # typed text. The line must survive with its full text.
        words = [
            (13, 71, 7, 10, 0.0, "1"),
            (26, 70, 62, 14, 0.0, "replaced"),
            (98, 70, 15, 14, 96.7, "by"),
            (122, 70, 30, 11, 96.6, "held"),
            (161, 69, 64, 12, 96.8, "modifier"),
        ]
        lines = _cluster_lines(words, 1.0)
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0].text, "1 replaced by held modifier")
        self.assertGreater(lines[0].conf, 55.0)

    def test_majority_garbage_row_stays_dropped(self):
        # The median must not resurrect rows that are mostly noise with a
        # couple of lucky words.
        words = [
            (10, 20, 10, 8, 7.0, "x"),
            (30, 20, 10, 8, 30.0, "o"),
            (50, 20, 10, 8, 35.0, "q"),
            (70, 20, 10, 8, 96.0, "a"),
        ]
        lines = _cluster_lines(words, 1.0)
        self.assertEqual([l for l in lines if l.conf >= 55.0], [])


class TestDeepRescue(unittest.TestCase):
    """Failed main-pass rows get a rescue pass at every window size.

    Regression history: the alt-confidence rescue pass ran only when the
    main-pass line was confident enough for readback-size crops; a maximized
    editor (1.6+ M px) dropped failed rows entirely on scan paths. Now the
    strip-based rescue runs whenever the main pass has failures, deep or
    not (a maximized Firefox page being the original motivation).
    """

    CROP_W, CROP_H = 2000, 800  # 1.6 M px: over the alt-pass budget

    def _run(self, deep: bool):
        crop = np.full((self.CROP_H, self.CROP_W, 3), 40, np.uint8)
        shapes = []

        def side_effect(gray, scale):
            shapes.append(gray.shape[:2])
            if gray.shape[:2] == (self.CROP_H, self.CROP_W):
                return [TextLine(x=100, y=100, w=1700, h=20, text="garbled", conf=10.0)]
            # The rescue pass now reads one composite image: strips stacked
            # with a white gap above and below each. A word's y is therefore
            # in that composite's coordinate space (half of its 2x pixels),
            # so 25 really means 25 + gap/2 here.
            lead = config.OCR_STRIP_GAP / 2.0
            return [TextLine(x=120, y=25 + lead, w=800, h=30,
                             text="replaced by held modifier", conf=92.0)]

        with unittest.mock.patch.object(Ocr, "_tesseract_words", side_effect=side_effect):
            out = Ocr().read(crop, deep=deep)
        return out, shapes

    def test_scan_path_rescues_when_main_pass_fails(self):
        out, shapes = self._run(deep=False)
        self.assertEqual([line.text for line in out], ["replaced by held modifier"])
        # strip-local (120, 25) + pad origin (40, 60), scale 1.0; the leading
        # composite gap must cancel out, so the answer is unchanged.
        self.assertEqual((out[0].x, out[0].y), (160, 85))
        self.assertGreaterEqual(len(shapes), 2)  # primary + strip pass ran

    def test_strip_rescue_is_one_composite_pass(self):
        """Many failed rows must not mean many tesseract processes.

        The per-strip rescue started one process (and loaded the model) per
        failed row: 22 rows measured 3.8 s for the few words it recovered.
        All strips now go through a single stacked image, so the whole rescue
        is exactly one extra call no matter how many rows failed.
        """
        crop = np.full((self.CROP_H, self.CROP_W, 3), 40, np.uint8)
        failed = [TextLine(x=100 + i * 40, y=100 + i * 30, w=300, h=20,
                           text=f"garbled{i}", conf=10.0) for i in range(6)]
        shapes = []

        def side_effect(gray, scale):
            shapes.append(gray.shape[:2])
            if gray.shape[:2] == (self.CROP_H, self.CROP_W):
                return failed
            return [TextLine(x=10, y=config.OCR_STRIP_GAP / 2.0 + 5, w=80, h=16,
                             text="recovered", conf=90.0)]

        with unittest.mock.patch.object(Ocr, "_tesseract_words", side_effect=side_effect):
            out = Ocr().read(crop, deep=True)
        self.assertEqual(len(shapes), 2, f"expected main+1 composite, got {shapes}")
        self.assertEqual([line.text for line in out], ["recovered"] * 1)

    def test_no_rescue_when_main_pass_is_clean(self):
        crop = np.full((self.CROP_H, self.CROP_W, 3), 40, np.uint8)
        good = [TextLine(x=100, y=100, w=200, h=20, text="good line", conf=92.0)]
        with unittest.mock.patch.object(Ocr, "_tesseract_words",
                                        return_value=good) as words, \
             unittest.mock.patch.object(Ocr, "_strip_rescue") as rescue:
            out = Ocr().read(crop, deep=False)
        self.assertEqual([line.text for line in out], ["good line"])
        words.assert_called_once()
        rescue.assert_not_called()


class TestCoordinates(unittest.TestCase):
    def test_permille_is_resolution_independent(self):
        self.assertEqual(_permille(0, config.LOGICAL_W), 0)
        self.assertEqual(_permille(config.LOGICAL_W, config.LOGICAL_W), 1000)
        self.assertEqual(_permille(config.LOGICAL_W / 2, config.LOGICAL_W), 500)

    def test_element_id_is_stable_and_content_addressable(self):
        box = [10, 20, 30, 40]
        self.assertEqual(_eid("t", box, "Save"), _eid("t", box, "Save"))
        self.assertNotEqual(_eid("t", box, "Save"), _eid("t", box, "Cancel"))
        self.assertNotEqual(_eid("t", box, "Save"), _eid("i", box, ""))

    def test_px_box_round_trips_through_permille(self):
        p = Perception.__new__(Perception)
        x, y, w, h = p._px_box([500, 250, 100, 50])
        self.assertEqual((x, y, w, h), (480, 135, 96, 27))


class TestDetectorDecode(unittest.TestCase):
    def _detector(self) -> Detector:
        d = Detector.__new__(Detector)
        d.conf, d.iou = 0.25, 0.45
        return d

    def test_decodes_xywh_rows_to_frame_boxes(self):
        # Real exports are (1, 5, 8400); keep enough candidates that the
        # transpose heuristic is not fooled by a tiny test tensor.
        out = np.zeros((1, 5, 10), np.float32)
        # One 64x64 box at the centre of a 640x640 letterbox, scale 1.0.
        out[0, :, 0] = [320, 320, 64, 64, 0.9]
        out[0, :, 1] = [100, 100, 20, 20, 0.1]  # below confidence
        boxes = self._detector()._decode(out, 1.0, 0, 0, (640, 640))
        self.assertEqual(len(boxes), 1)
        self.assertEqual((boxes[0].x, boxes[0].y, boxes[0].w, boxes[0].h), (288, 288, 64, 64))
        self.assertAlmostEqual(boxes[0].conf, 0.9, places=5)

    def test_handles_transposed_output(self):
        out = np.zeros((1, 10, 5), np.float32)  # (1, N, 5) rather than (1, 5, N)
        out[0, 0] = [320, 320, 40, 40, 0.8]
        boxes = self._detector()._decode(out, 1.0, 0, 0, (640, 640))
        self.assertEqual(len(boxes), 1)

    def test_letterbox_maps_back_to_original_coordinates(self):
        from smartcomp.detector import _letterbox

        img = np.zeros((540, 960, 3), np.uint8)
        blob, scale, pad_x, pad_y = _letterbox(img, 640)
        self.assertEqual(blob.shape, (1, 3, 640, 640))
        self.assertAlmostEqual(scale, 640 / 960)
        self.assertAlmostEqual(pad_y, (640 - 540 * (640 / 960)) / 2, places=5)
        self.assertEqual(pad_x, 0)


class TestXsetParse(unittest.TestCase):
    """The saved xset state must round-trip into the restore commands."""

    SAMPLE = """Keyboard Control:
  auto repeat:  on    key click percent:  0
LED mask:  00000000
Screen Saver:
  prefer blanking:  no    allow exposures:  no
  timeout:  900    cycle:  900
DPMS (Energy Star):
  Standby: 900    Suspend: 0    Off: 900
  DPMS is Enabled
"""

    def test_timeouts_are_extracted(self):
        state = _parse_xset(self.SAMPLE)
        self.assertEqual(state["ss_timeout"], 900)
        self.assertEqual(state["ss_cycle"], 900)
        self.assertEqual(state["dpms"], [900, 0, 900])
        self.assertTrue(state["dpms_enabled"])

    def test_dpms_disabled_is_recorded(self):
        state = _parse_xset(self.SAMPLE.replace("DPMS is Enabled", "DPMS is Disabled"))
        self.assertFalse(state["dpms_enabled"])

    def test_empty_output_does_not_raise(self):
        state = _parse_xset("")
        self.assertFalse(state["dpms_enabled"])
        self.assertNotIn("ss_timeout", state)


class TestHostLockGate(unittest.TestCase):
    """A locked host is never unlocked and never woken; the work moves instead.

    The daemon used to release the lock (ScreenSaver.SetActive) and force DPMS
    on before every action -- that is how an agent unlocked the user's desktop
    and kept it from locking. The gate is read-only: it reports, and desktop
    tools are re-run on the agent's own display (tools._LOCAL_ONLY lists what
    stays on the host).
    """

    def setUp(self):
        lock_mod._LOCK_CACHE = None
        self.addCleanup(setattr, lock_mod, "_LOCK_CACHE", None)

    def _reason(self, env: dict, locked: bool | None):
        base = {"DISPLAY": ":0", "SMARTCOMP_SESSION_NAME": "",
                "SMARTCOMP_HOST_LOCK_GATE": "1"}
        with unittest.mock.patch.dict(os.environ, {**base, **env}, clear=False), \
                unittest.mock.patch.object(lock_mod, "lock_state", return_value=locked):
            return lock_mod.blocked_reason()

    def test_locked_host_display_is_blocked(self):
        reason = self._reason({}, True)
        self.assertIsNotNone(reason)
        self.assertTrue(reason["locked"])
        self.assertIn("never unlocks", reason["error"])
        self.assertIn("session_create", reason["hint"])

    def test_unlocked_host_display_is_allowed(self):
        self.assertIsNone(self._reason({}, False))
        self.assertIsNone(self._reason({}, None))  # undetectable: do not block

    def test_session_display_is_never_blocked(self):
        self.assertIsNone(self._reason({"SMARTCOMP_SESSION_NAME": "alpha"}, True))

    def test_gate_can_be_disabled_explicitly(self):
        self.assertIsNone(self._reason({"SMARTCOMP_HOST_LOCK_GATE": "0"}, True))

    def test_gate_never_runs_a_command(self):
        # A refusal must not wake, inhibit or unlock: no subprocess at all.
        with unittest.mock.patch.dict(os.environ, {"DISPLAY": ":0",
                                                   "SMARTCOMP_SESSION_NAME": ""}), \
                unittest.mock.patch.object(lock_mod, "lock_state", return_value=True), \
                unittest.mock.patch.object(lock_mod, "is_locked", return_value=True), \
                unittest.mock.patch("subprocess.run") as run:
            self.assertIsNotNone(lock_mod.blocked_reason())
        run.assert_not_called()

    def test_desktop_tools_are_routed(self):
        # A tool that operates a desktop must move to the agent's display while
        # the host is locked, including the ones that used to be "not gated"
        # because EWMH still works through a lock: an agent on its own desktop
        # has to be able to start and inspect apps there.
        for name in ("click_element", "type_text", "screen_state", "get_changes",
                     "screenshot", "popups", "draw_stroke", "launch_app",
                     "list_windows", "focus_window", "move_window", "close_window",
                     "window_state", "clipboard_get", "clipboard_set", "app_status",
                     "wait_for_change"):
            self.assertNotIn(name, tools_mod._LOCAL_ONLY, name)

    def test_daemon_and_host_tools_are_local(self):
        # These belong to the daemon, the host or the process table -- routing
        # them to a virtual display would answer a different question.
        for name in ("health", "lock_state", "stats", "session_create",
                     "session_list", "session_endpoint", "session_close",
                     "session_align", "session_train", "session_end",
                     "remember", "recall", "run_skill", "skill_create",
                     "media_understand", "media_listen", "media_watch",
                     "app_owners", "app_release"):
            self.assertIn(name, tools_mod._LOCAL_ONLY, name)


class TestNoHostLockMutation(unittest.TestCase):
    """Regression guard: the lock/blank surface must stay read-only.

    The removed API is the point. If unlock, hard_unlock, ensure_screen,
    wake_display, inhibit or a lid-policy writer comes back, an action can
    silently take the user's lock away again.
    """

    def test_lock_module_has_no_unlock_api(self):
        for gone in ("unlock", "hard_unlock", "ensure_screen", "credential_path",
                     "read_credential", "credential_available"):
            self.assertFalse(hasattr(lock_mod, gone), gone)

    def test_display_module_only_reads(self):
        disp = display_mod.Display()
        for gone in ("wake_display", "enable", "disable", "apply_lock_policy",
                     "restore_lock_policy", "reassert"):
            self.assertFalse(hasattr(disp, gone), gone)

    def test_awake_module_is_gone(self):
        with self.assertRaises(ImportError):
            __import__("smartcomp.awake")


class TestMemoryErrorLog(unittest.TestCase):
    """Failed calls must record WHY they failed, not only that they failed.

    A live review found screenshot at 60/558 wins with nothing in the DB
    retaining the reason; the err column plus the stats() histogram is the fix.
    """

    def setUp(self):
        import tempfile

        self.dir = Path(tempfile.mkdtemp(prefix="smartcomp-test-"))
        patcher = unittest.mock.patch.object(config, "state_dir", lambda: self.dir)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.mem = memory_mod.Memory(path=self.dir / "mem.db")

    def test_failed_calls_record_their_error_text(self):
        self.mem.log(
            "scroll", {"amount": 400}, False, 24.0,
            error="TypeError: Input.scroll() got multiple values for argument 'axis'",
        )
        entry = next(e for e in self.mem.stats()["errors"] if e["tool"] == "scroll")
        self.assertIn("TypeError", entry["err"])
        self.assertEqual(entry["n"], 1)

    def test_ok_false_results_are_histogammed_too(self):
        self.mem.log("screenshot", {"window_id": "0x1a"}, False, 12.0,
                     error="BrokenPipeError: capture lost the X connection")
        self.mem.log("screenshot", {"window_id": "0x1a"}, False, 11.0,
                     error="BrokenPipeError: capture lost the X connection")
        entry = next(e for e in self.mem.stats()["errors"] if e["tool"] == "screenshot")
        self.assertEqual(entry["n"], 2)
        self.assertIn("BrokenPipeError", entry["err"])

    def test_successes_leave_the_histogram_empty(self):
        self.mem.log("click", {"x": 1}, True, 3.0)
        self.assertEqual(self.mem.stats()["errors"], [])

    def test_old_database_schema_is_migrated(self):
        # A DB created before the err/created/ttl columns must keep working.
        self.dir.joinpath("old.db").write_bytes(self.dir.joinpath("mem.db").read_bytes())
        fresh = memory_mod.Memory(path=self.dir / "old.db")
        fresh.log("t", {}, True, 1.0)
        fresh.remember("k", "v")
        self.assertEqual(fresh.recall("k")["value"], "v")


class TestObservationTTL(unittest.TestCase):
    """recall must report provenance: age, and staleness when a ttl expired."""

    def setUp(self):
        import tempfile

        self.dir = Path(tempfile.mkdtemp(prefix="smartcomp-test-"))
        patcher = unittest.mock.patch.object(config, "state_dir", lambda: self.dir)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.mem = memory_mod.Memory(path=self.dir / "mem.db")

    def test_recall_reports_the_age_of_a_fresh_fact(self):
        self.mem.remember("k1", "fresh", ttl_s=60)
        got = self.mem.recall("k1")
        self.assertEqual(got["value"], "fresh")
        self.assertFalse(got["stale"])
        self.assertLess(got["age_s"], 5)

    def test_expired_fact_is_stale_and_nulled(self):
        self.mem.remember("k2", "stale claim", ttl_s=5)
        self.mem.db.execute("UPDATE observations SET updated = updated - 30 WHERE key = 'k2'")
        self.mem.db.commit()
        got = self.mem.recall("k2")
        self.assertIsNone(got["value"])
        self.assertTrue(got["stale"])

    def test_no_ttl_means_facts_persist_but_age_is_reported(self):
        self.mem.remember("k3", {"binary": "firefox"})
        self.mem.db.execute("UPDATE observations SET updated = updated - 900 WHERE key = 'k3'")
        self.mem.db.commit()
        got = self.mem.recall("k3")
        self.assertEqual(got["value"], {"binary": "firefox"})
        self.assertFalse(got["stale"])
        self.assertGreater(got["age_s"], 890)

    def test_unknown_key_is_null_not_an_error(self):
        got = self.mem.recall("nope")
        self.assertIsNone(got["value"])
        self.assertIsNone(got["stale"])


class TestScrollClamp(unittest.TestCase):
    """scroll(amount=400) must not fire 400 wheel clicks."""

    def _input(self):
        inp = input_mod.Input.__new__(input_mod.Input)
        inp._d = unittest.mock.MagicMock()
        return inp

    def test_large_amount_is_capped(self):
        with unittest.mock.patch.object(input_mod.xtest, "fake_input") as fake:
            self._input().scroll(400)
        # 20 ticks x (press + release); no move since no x/y.
        self.assertEqual(fake.call_count, 2 * input_mod.MAX_SCROLL_TICKS)

    def test_normal_amount_is_uncapped(self):
        with unittest.mock.patch.object(input_mod.xtest, "fake_input") as fake:
            self._input().scroll(3, axis="horizontal", x=100, y=100)
        # 1 move + 3 ticks x (press + release).
        self.assertEqual(fake.call_count, 1 + 6)

    def test_horizontal_axis_is_case_insensitive(self):
        with unittest.mock.patch.object(input_mod.xtest, "fake_input") as fake:
            inp = self._input()
            inp.scroll(1, axis="Horizontal")
            inp.scroll(1, axis="horizontal")
        # 2 ticks x (press + release), every event on button 7 (right wheel).
        self.assertEqual(len(fake.call_args_list), 4)
        self.assertEqual({c.args[2] for c in fake.call_args_list}, {7})


class TestClipboardSet(unittest.TestCase):
    """xclip forks and owns the selection; our pipes must never wait for it."""

    def _proc(self, returncode=0):
        proc = unittest.mock.MagicMock()
        proc.returncode = returncode
        proc.communicate.return_value = (None, None)
        return proc

    def test_owner_gets_devnull_and_its_own_session(self):
        proc = self._proc()
        with unittest.mock.patch.object(
            input_mod.subprocess, "Popen", return_value=proc
        ) as popen:
            input_mod.Input.__new__(input_mod.Input).clipboard_set("hello")
        self.assertEqual(popen.call_args.kwargs["stdout"], subprocess.DEVNULL)
        self.assertEqual(popen.call_args.kwargs["stderr"], subprocess.DEVNULL)
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        proc.communicate.assert_called_once_with(b"hello", timeout=5)

    def test_stuck_xclip_is_killed_and_reported(self):
        proc = self._proc()
        proc.communicate.side_effect = subprocess.TimeoutExpired("xclip", 5)
        with unittest.mock.patch.object(input_mod.subprocess, "Popen", return_value=proc):
            with self.assertRaises(input_mod.InputError):
                input_mod.Input.__new__(input_mod.Input).clipboard_set("hello")
        proc.kill.assert_called_once()

    def test_nonzero_exit_is_reported(self):
        proc = self._proc(returncode=1)
        with unittest.mock.patch.object(input_mod.subprocess, "Popen", return_value=proc):
            with self.assertRaises(input_mod.InputError):
                input_mod.Input.__new__(input_mod.Input).clipboard_set("hello")


class TestDeltaCompaction(unittest.TestCase):
    """A window switch must not ship 200 deleted content-hash ids."""

    class _FakeFrame:
        native = np.zeros((540, 960, 3), np.uint8)
        bgr = np.zeros((540, 960, 3), np.uint8)
        capture_ms = 1.0

    class _FakeDiff:
        full = True
        tiles: list = []
        ms = 0.5

    class _FakeCapturer:
        native_size = (1920, 1080)

        def capture(self):
            return TestDeltaCompaction._FakeFrame()

        def diff(self, frame):
            return TestDeltaCompaction._FakeDiff()

        def reset(self):
            pass

    def _perception(self, n_elements: int):
        p = Perception.__new__(Perception)
        p.capturer = self._FakeCapturer()
        p.ocr = unittest.mock.MagicMock()
        p.ocr.last_ms = 0.0
        p.detector = unittest.mock.MagicMock()
        p.detector.ready = False
        p.seq = 0
        p.last_stats = {}
        p._recognize = lambda *a, **k: []
        p.elements = {}
        for i in range(n_elements):
            box = [0, i, 10, 10]
            eid = _eid("t", box, f"el{i}")
            p.elements[eid] = {"id": eid, "k": "t", "b": box, "x": f"el{i}",
                               "c": 0.9, "u": 1}
        return p

    def test_large_del_list_is_capped_with_a_count(self):
        payload = self._perception(20).get_changes()
        self.assertTrue(payload["full"])
        self.assertEqual(len(payload["del"]), 8)
        self.assertEqual(payload["del_dropped"], 12)

    def test_small_del_list_ships_in_full(self):
        payload = self._perception(5).get_changes()
        self.assertEqual(len(payload["del"]), 5)
        self.assertNotIn("del_dropped", payload)

    def test_verbose_del_still_returns_every_id(self):
        payload = self._perception(20).get_changes(verbose_del=True)
        self.assertEqual(len(payload["del"]), 20)
        self.assertNotIn("del_dropped", payload)


class TestClipBox(unittest.TestCase):
    """Window screenshots must clamp to the root or XGetImage BadMatches."""

    BOUNDS = {"left": 0, "top": 0, "width": 1920, "height": 1080}

    def test_inside_bounds_is_untouched(self):
        box = {"left": 10, "top": 10, "width": 400, "height": 300}
        self.assertEqual(_clip_box(box, self.BOUNDS), box)

    def test_negative_origin_is_clamped(self):
        box = {"left": -100, "top": -50, "width": 400, "height": 300}
        self.assertEqual(_clip_box(box, self.BOUNDS),
                         {"left": 0, "top": 0, "width": 300, "height": 250})

    def test_offscreen_edge_is_clamped(self):
        box = {"left": 1800, "top": 1000, "width": 400, "height": 300}
        self.assertEqual(_clip_box(box, self.BOUNDS),
                         {"left": 1800, "top": 1000, "width": 120, "height": 80})

    def test_entirely_offscreen_is_none(self):
        self.assertIsNone(
            _clip_box({"left": -400, "top": 100, "width": 300, "height": 200},
                      self.BOUNDS))
        self.assertIsNone(
            _clip_box({"left": 2000, "top": 100, "width": 300, "height": 200},
                      self.BOUNDS))


class TestDegenerateRead(unittest.TestCase):
    """A near-empty OCR read of a big window must be flagged, not trusted."""

    def test_firefox_single_line_read_is_flagged(self):
        # Real read from the reviewed session: tesseract returned "ao x".
        self.assertTrue(_looks_degenerate([{"x": "ao x"}], 1920, 1000))

    def test_live_firefox_sparse_read_is_flagged(self):
        # Real readback from the fix session: 5 fragments, 24 chars over a
        # 1.2 M px window. The first heuristic (lines < 3) missed it.
        lines = [{"x": "a x"}, {"x": "- a x"}, {"x": "fal"},
                 {"x": "x] +"}, {"x": "| New Tab"}]
        self.assertTrue(_looks_degenerate(lines, 1209, 1044))

    def test_empty_read_of_a_big_window_is_flagged(self):
        self.assertTrue(_looks_degenerate([], 1920, 1000))

    def test_normal_read_is_not_flagged(self):
        # A realistic page read: plenty of lines, plenty of characters.
        lines = [{"x": f"menu item {i} with a few words of text"} for i in range(12)]
        self.assertFalse(_looks_degenerate(lines, 1920, 1000))

    def test_small_window_with_few_lines_is_not_flagged(self):
        self.assertFalse(_looks_degenerate([{"x": "ok"}], 300, 200))


class TestStatsRecentWindow(unittest.TestCase):
    """Lifetime rates of a fixed bug must not read as current (screenshot 11%)."""

    def setUp(self):
        import tempfile

        self.dir = Path(tempfile.mkdtemp(prefix="smartcomp-test-"))
        patcher = unittest.mock.patch.object(config, "state_dir", lambda: self.dir)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.mem = memory_mod.Memory(path=self.dir / "mem.db")

    def test_old_failures_stay_out_of_the_recent_window(self):
        self.mem.log("screenshot", {}, True, 10.0)
        self.mem.log("screenshot", {}, False, 10.0, error="XError: BadMatch")
        self.mem.db.execute("UPDATE actions SET ts = ts - 3 * 86400 WHERE ok = 0")
        self.mem.db.commit()
        recent = self.mem.stats(hours=24)["recent"]["tools"]["screenshot"]
        self.assertEqual(recent["calls"], 1)
        self.assertEqual(recent["wins"], 1)
        self.assertEqual(recent["success_rate"], 1.0)
        lifetime = self.mem.stats(hours=0)["tools"]["screenshot"]
        self.assertEqual(lifetime["calls"], 2)
        self.assertEqual(lifetime["wins"], 1)

    def test_hours_zero_disables_the_window(self):
        self.mem.log("click", {}, True, 1.0)
        stats = self.mem.stats(hours=0)
        self.assertEqual(stats["recent"]["window_h"], 0)
        self.assertEqual(stats["recent"]["tools"], {})


class TestSessionRotation(unittest.TestCase):
    """An idle gap must start a new session instead of blending tasks.

    Observed live: clipboard probes from a fix-verification run appeared in
    a later browser-research session's end-of-session trace.
    """

    def setUp(self):
        import tempfile

        self.dir = Path(tempfile.mkdtemp(prefix="smartcomp-test-"))
        patcher = unittest.mock.patch.object(config, "state_dir", lambda: self.dir)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.mem = memory_mod.Memory(path=self.dir / "mem.db")

    def test_idle_gap_rotates_the_session(self):
        self.mem.idle_gap_s = 60.0
        self.mem.log("click", {}, True, 1.0)
        first = self.mem.session
        self.mem.last_action_ts = time.time() - 3600  # an hour passed
        self.mem.log("type_text", {}, True, 2.0)
        self.assertNotEqual(self.mem.session, first)
        rows = self.mem.db.execute(
            "SELECT session, tool FROM actions ORDER BY id").fetchall()
        self.assertEqual(rows[0]["tool"], "click")
        self.assertEqual(rows[0]["session"], first)
        self.assertEqual(rows[1]["tool"], "type_text")
        self.assertNotEqual(rows[1]["session"], first)

    def test_no_gap_keeps_one_session(self):
        self.mem.idle_gap_s = 60.0
        self.mem.log("click", {}, True, 1.0)
        first = self.mem.session
        self.mem.log("type_text", {}, True, 2.0)  # seconds later
        self.assertEqual(self.mem.session, first)

    def test_rotation_can_be_disabled(self):
        self.mem.idle_gap_s = None
        self.mem.log("click", {}, True, 1.0)
        first = self.mem.session
        self.mem.last_action_ts = time.time() - 999999
        self.mem.log("type_text", {}, True, 2.0)
        self.assertEqual(self.mem.session, first)


class TestWindowMatchAndCensus(unittest.TestCase):
    """Resume/census matching runs on identity fields alone, no X needed."""

    def _wins(self):
        return [
            {"id": "0x1", "title": "Mozilla Firefox", "class": "", "instance": "Navigator",
             "pid": 11, "comm": "firefox-esr", "cmd": "firefox-esr", "active": True},
            {"id": "0x2", "title": "Terminal", "class": "", "instance": "qterminal",
             "pid": 22, "comm": "qterminal", "cmd": "qterminal", "active": False},
            {"id": "0x3", "title": "About", "class": "", "instance": "Navigator",
             "pid": 11, "comm": "firefox-esr", "cmd": "firefox-esr", "active": False},
        ]

    def test_binary_matches_esr_style_variants(self):
        self.assertTrue(apps_mod.window_matches(self._wins()[0], "firefox"))
        self.assertFalse(apps_mod.window_matches(self._wins()[1], "firefox"))

    def test_resume_prefers_the_active_window(self):
        self.assertEqual(apps_mod.resume_target(self._wins(), "firefox")["id"], "0x1")
        self.assertIsNone(apps_mod.resume_target(self._wins(), "chromium"))

    def test_census_counts_processes_windows_and_focus(self):
        with unittest.mock.patch.object(
            apps_mod, "processes", lambda binary: [{"pid": 11}, {"pid": 99}]
        ):
            c = apps_mod.census(self._wins())
        ff = c["firefox"]
        self.assertTrue(ff["running"])
        self.assertEqual(ff["processes"], 2)
        self.assertEqual(len(ff["windows"]), 2)
        self.assertEqual(ff["focused"], "0x1")
        self.assertEqual(c["terminal"]["focused"], None)

    def test_tabs_are_gated_on_a_live_process(self):
        # Measured live: a daemon-restart kill left a 2-minute-old
        # recovery.jsonlz4 behind and the census answered "3 tabs open"
        # while running was false -- a fact about a browser nothing can see.
        with unittest.mock.patch.object(apps_mod, "processes", lambda b: []), \
             unittest.mock.patch.object(apps_mod, "firefox_tabs",
                                        lambda: {"total_tabs": 3}):
            c = apps_mod.census(self._wins())
        self.assertIsNone(c["firefox"]["tabs"])

    def test_tabs_read_when_running(self):
        with unittest.mock.patch.object(
            apps_mod, "processes", lambda b: [{"pid": 11}, {"pid": 99}]
        ), unittest.mock.patch.object(
            apps_mod, "firefox_tabs", lambda: {"total_tabs": 3, "windows": []}
        ):
            c = apps_mod.census(self._wins())
        self.assertEqual(c["firefox"]["tabs"]["total_tabs"], 3)


class TestTabSourceFreshness(unittest.TestCase):
    """A stale post-quit snapshot must not answer "how many tabs are open"."""

    def _file(self, name: str, age_s: float) -> Path:
        import tempfile

        path = Path(tempfile.mkdtemp(prefix="smartcomp-test-")) / name
        path.write_bytes(b"x")
        stamp = time.time() - age_s
        os.utime(path, (stamp, stamp))
        return path

    def test_recovery_is_always_usable(self):
        self.assertTrue(apps_mod._fresh_enough(self._file("recovery.jsonlz4", 5000), time.time()))

    def test_recent_previous_snapshot_is_usable(self):
        self.assertTrue(apps_mod._fresh_enough(self._file("previous.jsonlz4", 60), time.time()))

    def test_old_previous_snapshot_is_not(self):
        # Measured live: a 25-minute-old previous.jsonlz4 said 1 tab while
        # 3 were open. It describes a session that is over.
        self.assertFalse(apps_mod._fresh_enough(self._file("previous.jsonlz4", 1500), time.time()))


class TestProcessScan(unittest.TestCase):
    """processes() finds real pids by executable name, not by grep accident."""

    def test_finds_real_sleep_processes(self):
        procs = [subprocess.Popen(["sleep", f"3{i}"]) for i in range(2)]
        try:
            found = {p["pid"] for p in apps_mod.processes("sleep")}
            for proc in procs:
                self.assertIn(proc.pid, found)
        finally:
            for proc in procs:
                proc.kill()
            for proc in procs:
                proc.wait()

    def test_does_not_match_processes_merely_mentioning_the_name(self):
        # comm/cmd basename rule: a shell or grep mentioning "sleep" in its
        # argv is not "sleep". This test process itself is the counterexample.
        found = {p["pid"] for p in apps_mod.processes("unittest")}
        self.assertNotIn(os.getpid(), found)

    def test_zombies_are_not_app_instances(self):
        # Found live: a defunct firefox-esr passed as a real instance and the
        # resume guard refused a launch that should have happened.
        proc = subprocess.Popen(["sleep", "39"])
        proc.kill()  # a zombie until someone wait()s on it
        time.sleep(0.2)
        try:
            self.assertEqual(apps_mod._proc_state(proc.pid), "Z")
            self.assertNotIn(proc.pid,
                             [p["pid"] for p in apps_mod.processes("sleep")])
        finally:
            proc.wait()  # reap it


class TestMozLz4Decode(unittest.TestCase):
    """The Firefox session-store container round-trips through the decoder."""

    def test_decode_roundtrip(self):
        import struct
        import tempfile

        if apps_mod._lz4 is None:
            self.skipTest("lz4 module not available")
        import lz4.block

        payload = b'{"windows": [{"tabs": [1, 2, 3]}]}'
        blob = (apps_mod._MOZLZ4_MAGIC + struct.pack("<I", len(payload))
                + lz4.block.compress(payload, store_size=False))
        path = Path(tempfile.mkdtemp(prefix="smartcomp-test-")) / "recovery.jsonlz4"
        path.write_bytes(blob)
        self.assertEqual(apps_mod._mozlz4_decode(path), payload)

    def test_rejects_non_mozlz4_files(self):
        import struct
        import tempfile

        path = Path(tempfile.mkdtemp(prefix="smartcomp-test-")) / "f.jsonlz4"
        path.write_bytes(b"PK\x03\x04" + struct.pack("<I", 5) + b"12345")
        with self.assertRaises(ValueError):
            apps_mod._mozlz4_decode(path)


class TestSessionstoreParse(unittest.TestCase):
    """Tab counts and the selected tab's title come out of sessionstore JSON."""

    def test_tab_counts_and_selected_title(self):
        obj = {"windows": [
            {"selected": 2, "tabs": [
                {"entries": [{"title": "Kali Linux"}], "index": 1},
                {"entries": [{"title": "First"}, {"title": "SmartComp docs"}], "index": 2},
            ]},
            {"selected": 1, "tabs": [{"entries": [{"title": "New Tab"}], "index": 1}]},
        ]}
        parsed = apps_mod._parse_sessionstore(obj)
        self.assertEqual(parsed["total_tabs"], 3)
        self.assertEqual(parsed["windows"][0]["tabs"], 2)
        self.assertEqual(parsed["windows"][0]["selected_title"], "SmartComp docs")
        self.assertEqual(parsed["windows"][1]["selected_title"], "New Tab")

    def test_empty_sessionstore_is_safe(self):
        parsed = apps_mod._parse_sessionstore({})
        self.assertEqual(parsed["total_tabs"], 0)
        self.assertEqual(parsed["windows"], [])


class TestKeyAliases(unittest.TestCase):
    """key() must accept the vocabulary agents actually write."""

    def test_common_natural_names_resolve(self):
        for natural, canonical in [("backspace", "BackSpace"), ("esc", "Escape"),
                                   ("enter", "Return"), ("pgdn", "Page_Down"),
                                   ("pagedown", "Page_Down"), ("del", "Delete")]:
            want = XK.string_to_keysym(canonical)
            self.assertEqual(input_mod.Input._keysym(natural), want, natural)

    def test_canonical_names_still_resolve(self):
        self.assertEqual(input_mod.Input._keysym("BackSpace"),
                         XK.string_to_keysym("BackSpace"))
        self.assertEqual(input_mod.Input._keysym("Return"),
                         XK.string_to_keysym("Return"))

    def test_case_insensitive_sweep(self):
        # Any keysym written in any case resolves: "page_up" is not X11
        # canonical ("Page_Up") but must not be an "unknown key".
        self.assertEqual(input_mod.Input._keysym("page_up"),
                         XK.string_to_keysym("Page_Up"))

    def test_single_char_falls_back_to_ordinal(self):
        self.assertEqual(input_mod.Input._keysym("a"), ord("a"))

    def test_unknown_word_is_zero(self):
        self.assertEqual(input_mod.Input._keysym("notakey"), 0)


class TestResolveElement(unittest.TestCase):
    """Label/id addressing: the agent speaks in WHAT, smartcomp resolves WHERE."""

    def setUp(self):
        self.perception = unittest.mock.Mock()
        self.perception.elements = {
            "t1": {"id": "t1", "k": "t", "b": [100, 100, 200, 20],
                   "x": "Paste", "c": 0.9, "u": 1},
            "t2": {"id": "t2", "k": "t", "b": [100, 200, 300, 20],
                   "x": "Paste Special", "c": 0.9, "u": 1},
            "i1": {"id": "i1", "k": "i", "b": [500, 500, 40, 40], "c": 0.8},
        }

    def test_exact_label_beats_substring(self):
        el, err = tools_mod._resolve_element(self.perception, None, "paste")
        self.assertIsNone(err)  # exact "Paste" wins over "Paste Special"

    def test_specific_label_resolves_to_one(self):
        el, err = tools_mod._resolve_element(self.perception, None, "paste special")
        self.assertEqual(el["id"], "t2")

    def test_unknown_label_lists_visible_text(self):
        _, err = tools_mod._resolve_element(self.perception, None, "Accept")
        self.assertEqual(err["ok"], False)
        self.assertIn("Paste", err["visible_text"][0])

    def test_icons_have_no_label_and_are_said_so(self):
        _, err = tools_mod._resolve_element(self.perception, None, "hamburger")
        self.assertIn("kind i", err["hint"])

    def test_unknown_id_names_the_id(self):
        _, err = tools_mod._resolve_element(self.perception, "zz9", None)
        self.assertIn("zz9", err["error"])

    def test_case_insensitive(self):
        el, err = tools_mod._resolve_element(self.perception, None, "PASTE")
        self.assertIsNone(err)
        self.assertEqual(el["id"], "t1")


class TestActionReports(unittest.TestCase):
    """Action results carry what the action did to the screen, as element ids."""

    def setUp(self):
        self.perception = unittest.mock.Mock()
        self.perception.get_changes.return_value = {
            "seq": 7, "full": False,
            "add": [
                {"id": "a1", "k": "t", "b": [10, 20, 100, 20], "x": "Save changes?",
                 "c": 0.9, "u": 1},
                {"id": "a2", "k": "i", "b": [30, 40, 50, 40], "c": 0.8},
            ],
            "del": ["gone1"],
            "stats": {"total": 412.5},
        }

    def test_report_trims_to_actable_fields(self):
        result = tools_mod._report_delta(self.perception, {"ok": True})
        changed = result["changed"]
        self.assertEqual(changed["add"][0],
                         {"id": "a1", "k": "t", "center": [60, 30],
                          "text": "Save changes?"})
        self.assertNotIn("b", changed["add"][0])  # geometry is not the agent's job
        self.assertEqual(changed["del"], ["gone1"])
        self.assertEqual(changed["ms"], 412.5)

    def test_full_capture_is_not_a_flood(self):
        self.perception.get_changes.return_value = {
            "seq": 8, "full": True,
            "add": [{"id": f"a{i}", "k": "t", "b": [0, i, 10, 10],
                     "x": "t", "c": 0.9} for i in range(50)],
            "stats": {"total": 5000.0},
        }
        result = tools_mod._report_delta(self.perception, {"ok": True})
        self.assertNotIn("add", result["changed"])
        self.assertIn("baseline", result["changed"]["note"])

    def test_adds_are_capped(self):
        self.perception.get_changes.return_value = {
            "seq": 9, "full": False,
            "add": [{"id": f"a{i}", "k": "t", "b": [0, i, 10, 10], "x": "x",
                     "c": 0.9} for i in range(12)],
            "stats": {},
        }
        result = tools_mod._report_delta(self.perception, {"ok": True})
        self.assertEqual(len(result["changed"]["add"]), 8)
        self.assertEqual(result["changed"]["add_dropped"], 4)

    def test_adds_lead_with_what_is_nearest_the_action(self):
        # Measured live: a click that opened a desktop menu while a terminal
        # behind it scrolled reported the scrollback lines first (OCR order)
        # and the menu fell into add_dropped. The report answers "what did
        # MY action bring up", so distance from the action orders it.
        far = {"id": "tui", "k": "t", "b": [0, 140, 300, 20], "x": "scrollback",
               "c": 0.9}
        near = {"id": "menu", "k": "t", "b": [860, 740, 120, 20],
                "x": "Open Terminal Here", "c": 0.9}
        self.perception.get_changes.return_value = {
            "seq": 11, "full": False, "add": [far, near],
            "stats": {"tiles": 2},
        }
        result = tools_mod._report_delta(self.perception, {"ok": True},
                                         at=[920, 750])
        self.assertEqual(result["changed"]["add"][0]["id"], "menu")
        self.assertEqual(result["changed"]["add"][1]["id"], "tui")

    def test_idle_report_is_small(self):
        self.perception.get_changes.return_value = {"seq": 10, "full": False,
                                                    "stats": {"total": 51.2}}
        result = tools_mod._report_delta(self.perception, {"ok": True})
        self.assertEqual(set(result["changed"]), {"seq", "full", "ms"})

    def test_half_drawn_popup_is_retried_once(self):
        # Measured live: a right-click menu open -- the action's pass OCR'd
        # the background-only paint (tiles changed, zero elements) and the
        # popup was invisible in the very call that opened it.
        self.perception.get_changes.side_effect = [
            {"seq": 1, "full": False, "stats": {"tiles": 3, "total": 4000.0}},
            {"seq": 2, "full": False,
             "add": [{"id": "m1", "k": "t", "b": [860, 730, 120, 20],
                      "x": "Open Terminal Here", "c": 0.9}],
             "stats": {"tiles": 2, "total": 900.0}},
        ]
        result = tools_mod._report_delta(self.perception, {"ok": True})
        self.assertEqual(self.perception.get_changes.call_count, 2)
        self.assertEqual(result["changed"]["add"][0]["text"], "Open Terminal Here")

    def test_no_retry_when_the_first_pass_answered(self):
        self.perception.get_changes.return_value = {
            "seq": 1, "full": False,
            "add": [{"id": "a1", "k": "t", "b": [0, 0, 10, 10], "x": "t",
                     "c": 0.9}],
            "stats": {"tiles": 1},
        }
        tools_mod._report_delta(self.perception, {"ok": True})
        self.assertEqual(self.perception.get_changes.call_count, 1)

    def test_late_dialog_beside_noise_is_retried_until_near(self):
        # Measured live: close_window's report pass returned Firefox noise
        # while the save dialog painted one beat later; the old retry
        # skipped because SOME adds existed. The answer to "what did MY
        # close do" was missing from the point of view of the action.
        far = {"id": "ff", "k": "t", "b": [100, 100, 100, 20], "x": "page",
               "c": 0.9}
        near = {"id": "dlg", "k": "t", "b": [500, 700, 120, 20],
                "x": "Don't Save", "c": 0.9}
        self.perception.get_changes.side_effect = [
            {"seq": 1, "full": False, "add": [far], "stats": {"tiles": 4}},
            {"seq": 2, "full": False, "add": [far, near], "stats": {"tiles": 2}},
        ]
        result = tools_mod._report_delta(self.perception, {"ok": True},
                                         at=[560, 700])
        self.assertEqual(self.perception.get_changes.call_count, 2)
        self.assertEqual(result["changed"]["add"][0]["id"], "dlg")

    def test_no_retry_when_an_add_is_already_near(self):
        near = {"id": "dlg", "k": "t", "b": [500, 700, 120, 20],
                "x": "Don't Save", "c": 0.9}
        self.perception.get_changes.return_value = {
            "seq": 1, "full": False, "add": [near], "stats": {"tiles": 1},
        }
        tools_mod._report_delta(self.perception, {"ok": True}, at=[520, 710])
        self.assertEqual(self.perception.get_changes.call_count, 1)

    def test_silent_action_point_waits_a_beat_for_the_late_paint(self):
        # Measured live: close_window's report pass ran 55 ms after the
        # close on a fully idle screen; the prompt (an IPC round trip away)
        # painted after it, so "no motion" and "no retry" were both wrong.
        silent = {"seq": 1, "full": False, "stats": {"tiles": 0}}
        dialog = {"id": "dlg", "k": "t", "b": [500, 700, 120, 20],
                  "x": "Don't Save", "c": 0.9}
        self.perception.get_changes.side_effect = [
            silent,
            {"seq": 2, "full": False, "add": [dialog], "stats": {"tiles": 2}},
        ]
        with unittest.mock.patch.object(tools_mod.time, "sleep") as slept:
            result = tools_mod._report_delta(self.perception, {"ok": True},
                                             at=[520, 710])
        slept.assert_called_once()
        self.assertEqual(self.perception.get_changes.call_count, 2)
        self.assertEqual(result["changed"]["add"][0]["text"], "Don't Save")

    def test_idle_screen_without_action_point_does_not_wait(self):
        # describe/get_changes style polls on a still screen must stay
        # cheap: the late-paint wait only applies when an action point
        # says an answer is expected.
        self.perception.get_changes.return_value = {
            "seq": 1, "full": False, "stats": {"tiles": 0},
        }
        with unittest.mock.patch.object(tools_mod.time, "sleep") as slept:
            tools_mod._report_delta(self.perception, {"ok": True})
        slept.assert_not_called()
        self.assertEqual(self.perception.get_changes.call_count, 1)

    def test_no_retry_on_idle_or_full_passes(self):
        self.perception.get_changes.return_value = {
            "seq": 1, "full": False, "stats": {"tiles": 0},
        }
        tools_mod._report_delta(self.perception, {"ok": True})
        self.assertEqual(self.perception.get_changes.call_count, 1)
        self.perception.get_changes.return_value = {
            "seq": 1, "full": True, "stats": {"tiles": 9},
        }
        tools_mod._report_delta(self.perception, {"ok": True})
        self.assertEqual(self.perception.get_changes.call_count, 2)  # setUp call + this one


class TestDescribeScreen(unittest.TestCase):
    """describe_screen's parts: colors and the plain-JSON composition."""

    def test_dominant_colors_hex_and_share(self):
        img = np.full((40, 40, 3), 40, dtype=np.uint8)  # BGR dark gray theme
        img[10:20, 10:30] = (32, 32, 200)  # a red patch (B=32, G=32, R=200)
        colors = tools_mod._dominant_colors(img)
        self.assertEqual(colors[0]["hex"], "#202020")  # 87.5% of pixels
        self.assertEqual(colors[1]["hex"], "#c02020")  # the red patch
        self.assertAlmostEqual(
            sum(c["share"] for c in colors), 1.0, places=2)
        shares = [c["share"] for c in colors]
        self.assertEqual(shares, sorted(shares, reverse=True))

    def test_dominant_colors_noise_is_quantized_into_themes(self):
        # A flat theme with +-3 dithering noise must read as ONE color.
        rng = np.random.default_rng(7)
        img = np.full((64, 64, 3), 36, dtype=np.uint8) + rng.integers(
            -3, 4, size=(64, 64, 3), dtype=np.int16).astype(np.uint8)
        self.assertEqual(tools_mod._dominant_colors(img, limit=3)[0]["hex"],
                         "#202020")

    def _window(self, **over):
        win = {"id": "0x1", "comm": "mousepad", "title": "untitled - mousepad",
               "x": 100, "y": 100, "w": 800, "h": 600, "active": True}
        win.update(over)
        return win

    def test_controls_are_the_active_windows_elements(self):
        inside = {"id": "t1", "k": "t", "b": [100, 200, 150, 30],
                  "x": "File", "c": 0.9, "u": 1}
        outside = {"id": "t2", "k": "t", "b": [900, 200, 50, 20],
                   "x": "other app", "c": 0.9, "u": 1}
        out = tools_mod._compose_description(
            self._window(), [self._window()],
            [outside, inside], [{"hex": "#202020", "share": 1.0}],
            [1920, 1080])
        self.assertEqual(out["active"]["app"], "mousepad")
        self.assertEqual([c["id"] for c in out["controls"]], ["t1"])
        self.assertEqual(out["screen"], [1920, 1080])

    def test_controls_read_top_to_bottom(self):
        lower = {"id": "low", "k": "t", "b": [100, 500, 150, 30], "x": "z",
                 "c": 0.9}
        upper = {"id": "up", "k": "t", "b": [100, 200, 150, 30], "x": "a",
                 "c": 0.9}
        out = tools_mod._compose_description(
            self._window(), [self._window()], [lower, upper], [], [1920, 1080])
        self.assertEqual([c["id"] for c in out["controls"]], ["up", "low"])

    def test_controls_are_capped_with_a_count(self):
        # All 15 inside the window's permille box (y 93..649): centers at
        # 97 + i*35 all land inside; x 100 is inside [52, 469].
        els = [{"id": f"e{i}", "k": "t", "b": [50, 90 + i * 35, 100, 15],
                "x": "t", "c": 0.9} for i in range(15)]
        out = tools_mod._compose_description(
            self._window(), [self._window()], els, [], [1920, 1080])
        self.assertEqual(len(out["controls"]), 12)
        self.assertEqual(out["controls_dropped"], 3)
        self.assertIn("read_window", out["hint"])

    def test_no_active_window_means_no_controls(self):
        out = tools_mod._compose_description(
            None, [], [], [], [1920, 1080])
        self.assertIsNone(out["active"])
        self.assertNotIn("controls", out)

    def test_other_windows_capped(self):
        wins = [self._window()] + [self._window(id=f"0x{i}", comm=f"app{i}",
                                                active=False)
                                   for i in range(2, 10)]
        out = tools_mod._compose_description(
            self._window(), wins, [], [], [1920, 1080])
        self.assertEqual(len(out["other_windows"]), 6)
        self.assertEqual(out["other_windows_dropped"], 2)


    def test_blank_spot_avoids_elements(self):
        wins = unittest.mock.Mock()
        wins.list.return_value = [{"id": "w", "active": True,
                                    "x": 0, "y": 0, "w": 960, "h": 540}]
        p = unittest.mock.Mock()
        p.capturer.native_size = (960, 540)
        p.elements = {"t": {"id": "t", "k": "t", "b": [400, 400, 100, 20],
                            "x": "button"}}
        spot = tools_mod._blank_spot(wins, p)
        self.assertNotEqual(spot, [450, 410])  # not the element's center
        self.assertTrue(0 <= spot[0] <= 1000 and 0 <= spot[1] <= 1000)


    def test_tiny_glyph_snaps_to_overlapping_icon(self):
        p = unittest.mock.Mock()
        glyph = {"id": "g", "k": "t", "b": [966, 189, 5, 9], "x": "x"}
        icon = {"id": "i", "k": "i", "b": [959, 180, 16, 22], "c": 0.44}
        p.elements = {"g": glyph, "i": icon}
        self.assertEqual(tools_mod._click_box(p, glyph), icon["b"])

    def test_large_element_unchanged(self):
        p = unittest.mock.Mock()
        btn = {"id": "b", "k": "t", "b": [100, 100, 200, 40], "x": "Save"}
        p.elements = {"b": btn}
        self.assertEqual(tools_mod._click_box(p, btn), btn["b"])


class TestCloseWindowReport(unittest.TestCase):
    """close() is the action whose answer IS the prompt: it arrives in-call."""

    def setUp(self):
        mcp = unittest.mock.Mock()
        mcp.tool.return_value = lambda fn: fn
        self.wins = unittest.mock.Mock()
        self.perception = unittest.mock.Mock()
        self.perception.capturer.native_size = (1920, 1080)
        # Dialog elements near the closing window's center in permille.
        self.perception.get_changes.return_value = {
            "seq": 5, "full": False,
            "add": [{"id": "dlg", "k": "t", "b": [500, 700, 120, 20],
                     "x": "Don't Save", "c": 0.9}],
            "stats": {"tiles": 2},
        }
        # wins.list() geometry is native pixels (450, 350 -> ~234, 324).
        self.wins.list.return_value = [
            {"id": "0x9", "comm": "mousepad", "x": 200, "y": 150,
             "w": 500, "h": 400}]
        self.wins.close.return_value = {"ok": True}
        # These are host-desktop tests, so pin the host to "unlocked": the
        # wrapper consults the real lock, and a test suite must not change its
        # meaning depending on whether the operator has the screen locked.
        unlocked = unittest.mock.patch.object(lock_mod, "lock_state", return_value=False)
        unlocked.start()
        self.addCleanup(unlocked.stop)
        registry = tools_mod.register(
            mcp, self.perception, unittest.mock.Mock(), self.wins,
            unittest.mock.Mock(), display=unittest.mock.Mock())
        self.close_window = registry["close_window"]

    def test_dialog_arrives_in_the_close_result(self):
        out = self.close_window(window_id="0x9")
        self.assertIn("changed", out)
        self.assertEqual(out["changed"]["add"][0]["text"], "Don't Save")

    def test_no_report_when_disabled(self):
        out = self.close_window(window_id="0x9", report=False)
        self.assertNotIn("changed", out)

    def test_failed_close_stays_bare(self):
        self.wins.close.return_value = {"ok": False, "error": "no such window"}
        out = self.close_window(window_id="0x9")
        self.assertNotIn("changed", out)


class TestOwnership(unittest.TestCase):
    """Shared app-ownership: never resume another smartcomp's app."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="smartcomp-own-")
        os.environ["SMARTCOMP_STATE_DIR"] = self.tmp
        os.environ["SMARTCOMP_OWNERS_DIR"] = self.tmp
        os.environ["SMARTCOMP_SESSION_NAME"] = "alpha"

    def tearDown(self):
        os.environ.pop("SMARTCOMP_SESSION_NAME", None)
        os.environ.pop("SMARTCOMP_OWNERS_DIR", None)
        if "SMARTCOMP_STATE_DIR" in os.environ:
            del os.environ["SMARTCOMP_STATE_DIR"]
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_claim_and_owner_of_self_is_none(self):
        from smartcomp import ownership as own
        own.claim("firefox", "0x1", os.getpid())
        self.assertIsNone(own.owner_of("firefox"))  # our own: resume OK

    def test_other_session_sees_claim(self):
        from smartcomp import ownership as own
        own.claim("firefox", "0x1", os.getpid())
        os.environ["SMARTCOMP_SESSION_NAME"] = "beta"
        entry = own.owner_of("firefox")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["owner"], "alpha")

    def test_release_and_status(self):
        from smartcomp import ownership as own
        own.claim("firefox", "0x1", os.getpid())
        own.release("firefox")
        self.assertIsNone(own.owner_of("firefox"))
        self.assertIn("claims", own.status())


class TestSkillbook(unittest.TestCase):
    """Pre-made skills + self-built skill creation from work."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="smartcomp-skill-")
        from smartcomp.memory import Memory
        self.mem = Memory(path=Path(self.tmp) / "memory.db")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_premade_skills_installed(self):
        out = skillbook_mod.install_premade(self.mem)
        self.assertEqual(out["installed"], len(skillbook_mod.PREMADE))
        for name in ("builtin/watch-video", "builtin/listen-reel",
                     "builtin/research-tabs", "builtin/open-app"):
            self.assertIsNotNone(self.mem.skill(name))

    def test_premade_video_skill_has_media_steps(self):
        steps = skillbook_mod.PREMADE["watch-video"]["steps"]
        tools = [s["tool"] for s in steps]
        self.assertIn("media_understand", tools)
        self.assertIn("explain_screen", tools)

    def test_auto_skill_from_work(self):
        out = skillbook_mod.auto_skill(self.mem, "myflow", [
            {"tool": "click_element", "args": {"label": "OK"}}])
        self.assertTrue(out["ok"])
        s = self.mem.skill("self/myflow")
        self.assertEqual(s["steps"][0]["tool"], "click_element")

    def test_learn_from_recent(self):
        self.mem.log("click_element", {"label": "OK"}, True, 10)
        self.mem.log("type_text", {"text": "hi"}, True, 10)
        out = skillbook_mod.learn_from_recent(self.mem)
        self.assertTrue(out["ok"])
        self.assertGreaterEqual(out["steps"], 2)


class TestSessionReap(unittest.TestCase):
    """Closing a session must not leave a zombie per child process.

    Xvfb/WM/session-daemon are children of the daemon; SIGTERM without
    waitpid parks each one as <defunct> for the daemon's whole lifetime.
    """

    def test_killed_child_is_reaped(self):
        proc = subprocess.Popen(["sleep", "30"])
        os.kill(proc.pid, 15)
        sessions_mod._reap(proc.pid)
        self.assertFalse(Path(f"/proc/{proc.pid}").exists())
        proc.returncode = 0  # already reaped; keep Popen from waiting again

    def test_non_child_pid_is_a_noop(self):
        sessions_mod._reap(999_999_999, timeout=0.1)  # must not raise


class TestSessionManager(unittest.TestCase):
    """Per-session virtual displays: own cursor, shared training/memory."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="smartcomp-sessions-")
        self.mgr = sessions_mod.SessionManager.__new__(sessions_mod.SessionManager)
        self.mgr.root = Path(self.tmp) / "sessions"
        self.mgr.root.mkdir(parents=True)
        self.mgr.registry_path = self.mgr.root / "registry.json"
        self.mgr.shared = unittest.mock.Mock()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _rec(self, name, display, port):
        d = self.mgr.root / name
        d.mkdir(exist_ok=True)
        return {"display": display, "port": port, "pid": 1, "xvfb_pid": 2,
                "wm_pid": None, "dir": str(d), "created": 1.0}

    def test_free_display_skips_used_and_host(self):
        """Registry entries and live servers are skipped; stale locks are not.

        A killed Xvfb leaves its lock and socket behind. Counting those as
        taken leaked display numbers on every crash, so a dead owner pid means
        the display is free again. The socket directory is patched to a temp
        dir: the host's own displays must not decide the test's answer.
        """
        with tempfile.TemporaryDirectory() as xdir:
            with unittest.mock.patch.object(sessions_mod, "_X11_DIR", Path(xdir)):
                reg = {"sessions": {"a": self._rec("a", ":1000", 8792)}}
                Path(xdir, "X1001").touch()          # live server, lock is us
                Path("/tmp/.X1001-lock").write_text(str(os.getpid()))
                Path(xdir, "X1002").touch()          # stale: owner long gone
                Path("/tmp/.X1002-lock").write_text("999999")
                try:
                    self.assertEqual(self.mgr._free_display(reg), ":1002")
                finally:
                    Path("/tmp/.X1001-lock").unlink(missing_ok=True)
                    Path("/tmp/.X1002-lock").unlink(missing_ok=True)

    def test_session_creates_token_immediately(self):
        d = self.mgr.root / "tokentest"
        d.mkdir(exist_ok=True)
        token = self.mgr.ensure_token(d)
        self.assertTrue(token)
        self.assertEqual(self.mgr.ensure_token(d), token)  # stable
        self.assertEqual((d / "token").stat().st_mode & 0o777, 0o600)

    def test_recent_window_visible_to_second_reader(self):
        popups_mod._prev_windows = None
        popups_mod._recent_windows.clear()
        w = unittest.mock.Mock()
        win = {"id": "0x7", "comm": "firefox", "title": "Sign in - Google Accounts",
               "x": 100, "y": 100, "w": 720, "h": 680}
        w.list.return_value = [win]
        first = popups_mod.new_windows(w)           # baseline -> empty
        self.assertEqual(first, [])
        w.list.return_value = [win, {"id": "0x8", "comm": "firefox", "title": "Popup"}]
        second = popups_mod.new_windows(w)          # action report sees 0x8
        self.assertEqual([x["id"] for x in second], ["0x8"])
        # a following popups() call must still see it via recent memory
        self.assertEqual([x["id"] for x in popups_mod.recent_windows(w)], ["0x8"])

    def test_registry_round_trip(self):
        self.mgr._save({"sessions": {"a": self._rec("a", ":1000", 8792)}})
        self.assertIn("a", self.mgr._load()["sessions"])

    def test_close_removes_session_and_kills_processes(self):
        self.mgr._save({"sessions": {"a": self._rec("a", ":1000", 8792)}})
        with unittest.mock.patch.object(sessions_mod.os, "killpg"), \
             unittest.mock.patch.object(sessions_mod.os, "getpgid", return_value=7):
            out = self.mgr.close("a")
        self.assertTrue(out["ok"])
        self.assertNotIn("a", self.mgr._load()["sessions"])

    def test_align_merges_into_shared_pool(self):
        rec = self._rec("a", ":1000", 8792)
        self.mgr._save({"sessions": {"a": rec}})
        mem = unittest.mock.Mock()
        mem.skills.return_value = [{"name": "flow", "steps": [{"tool": "click"}],
                                    "description": "d"}]
        mem.observations.return_value = [{"key": "k", "value": "v"}]
        mem.skill.return_value = None
        self.mgr.shared.skills.return_value = []
        with unittest.mock.patch.object(self.mgr, "_session_memory", return_value=mem):
            out = self.mgr.align()
        self.assertEqual(out["pushed"], 2)
        self.assertTrue(self.mgr.shared.learn_skill.called)
        self.assertTrue(self.mgr.shared.remember.called)


class TestPopupDetection(unittest.TestCase):
    """A popup must surface as a first-class block, never buried in a delta."""

    def setUp(self):
        self.wins = unittest.mock.Mock()
        self.perception = unittest.mock.Mock()
        self.perception.capturer.native_size = (1920, 1080)
        self.perception.elements = {}
        popups_mod._prev_windows = None
        popups_mod._recent_windows.clear()
        # detect() also reads the live X override-redirect tree; a menu open
        # on the developer's desktop is not part of what these tests assert.
        self._x_patch = unittest.mock.patch.object(popups_mod, "_extra_x_windows",
                                                   return_value=[])
        self._x_patch.start()
        self.addCleanup(self._x_patch.stop)

    def test_dialog_window_detected(self):
        self.wins.list.return_value = [
            {"id": "0x1", "comm": "brave", "title": "Confirm close all tabs",
             "x": 700, "y": 300, "w": 500, "h": 200}]
        self.perception.elements = {}
        found = popups_mod.detect(self.wins, self.perception)
        self.assertTrue(any(f["source"] == "window" for f in found))

    def test_button_elements_detected(self):
        self.wins.list.return_value = [
            {"id": "0x1", "comm": "mousepad", "title": "Save Changes",
             "x": 184, "y": 789, "w": 452, "h": 113}]
        self.perception.capturer.native_size = (1920, 1080)
        # Buttons inside the popup rect (permille of native screen)
        self.perception.elements = {
            "a": {"id": "a", "k": "t", "b": [96, 735, 60, 20],
                  "x": "Close all tabs"},
            "b": {"id": "b", "k": "t", "b": [200, 760, 40, 20], "x": "Cancel"},
        }
        found = popups_mod.detect(self.wins, self.perception)
        out = popups_mod.summarize(found)
        self.assertGreaterEqual(out["count"], 1)
        self.assertEqual({b["text"] for b in out["buttons"]},
                         {"Close all tabs", "Cancel"})

    def test_x_tree_override_redirect_popup_detected(self):
        # A notification bubble is invisible to the WM client list; only the
        # full X tree shows it. This was the missed-popup bug.
        self.wins.list.return_value = []
        with unittest.mock.patch.object(popups_mod, "_extra_x_windows", return_value=[
            {"id": "0x2a00004", "title": "Notification", "app": "x-popup",
             "source": "x-tree"}]):
            found = popups_mod.detect(self.wins, self.perception)
        out = popups_mod.summarize(found)
        self.assertGreaterEqual(out["count"], 1)
        self.assertTrue(any("Notification" in w["title"] for w in out["windows"]))

    def test_inpage_banner_detected(self):
        # Firefox shows a warning INSIDE the page (no separate X window).
        self.wins.list.return_value = []
        self.perception.elements = {
            "w": {"id": "w", "k": "t", "b": [400, 100, 600, 30],
                  "x": "The bookmarks system will not be functional because a file is in use by another application"},
        }
        found = popups_mod.detect(self.wins, self.perception)
        out = popups_mod.summarize(found)
        self.assertGreaterEqual(out["count"], 1)
        self.assertTrue(any("not be functional" in b["text"] for b in out["banners"]))

    def test_new_window_detected_after_click(self):
        # The Google sign-in case: a new window with an unremarkable title
        # and a large size must still surface as a popup candidate.
        popups_mod._prev_windows = None
        first = unittest.mock.Mock()
        first.list.return_value = [{"id": "0x1", "comm": "firefox", "title": "Search",
                                    "x": 0, "y": 0, "w": 1200, "h": 900}]
        self.assertFalse(any(f["source"] == "new-window" for f in
                             popups_mod.detect(first, self.perception)))
        second = unittest.mock.Mock()
        second.list.return_value = first.list.return_value + [
            {"id": "0x2", "comm": "firefox", "title": "Sign in - Google Accounts",
             "x": 400, "y": 100, "w": 720, "h": 680}]
        found = popups_mod.detect(second, self.perception)
        new = [f for f in found if f["source"] == "new-window"]
        self.assertEqual(len(new), 1)
        self.assertEqual(new[0]["title"], "Sign in - Google Accounts")

    def test_new_window_summary_includes_it(self):
        popups_mod._prev_windows = {"0x1"}
        wins = unittest.mock.Mock()
        wins.list.return_value = [{"id": "0x1", "comm": "a", "title": "a"},
                                  {"id": "0x2", "comm": "firefox", "title": "Sign in"}]
        out = popups_mod.summarize(popups_mod.detect(wins, self.perception))
        self.assertTrue(any(w["id"] == "0x2" for w in out["windows"]))

    def test_empty_screen_no_popup(self):
        self.wins.list.return_value = [
            {"id": "0x1", "comm": "brave", "title": "Big normal page",
             "x": 0, "y": 0, "w": 1200, "h": 1000}]
        self.perception.elements = {}
        self.assertEqual(popups_mod.summarize(popups_mod.detect(
            self.wins, self.perception))["count"], 0)


class TestXWindowParsing(unittest.TestCase):
    """Menus/notifications from the X tree are popups; utility windows are not."""

    SAMPLE = """
     0x160011b "Whisker Menu": ("wrapper-2.0" "Xfce4-panel")  1920x1080+0+0  +0+0
     0x2000001 "xfce4-notifyd": ("xfce4-notifyd" "Xfce4-notifyd")  300x80+1000+30  +1000+30
     0x3000001 "": ("utility" "Utility")  1x1+-100+-100  +-100+-100
     0x4000001 "Firefox": ("firefox" "Firefox")  1200x900+700+50  +700+50
    """

    def test_parses_popup_like_windows(self):
        titles = [f["title"] for f in popups_mod._parse_x_windows(self.SAMPLE)]
        self.assertIn("Whisker Menu", titles)
        self.assertIn("xfce4-notifyd", titles)

    def test_skips_utility_and_normal_windows(self):
        found = popups_mod._parse_x_windows(self.SAMPLE)
        self.assertEqual(len(found), 2)
        self.assertTrue(all(f["source"] == "x-tree" for f in found))


class TestVisionDecider(unittest.TestCase):
    """The decider picks the explanation that repeats more of the other's
    distinctive words -- the more accurate reading of the same screen."""

    def test_mirage_picks_when_it_covers_the_words(self):
        v = vision_mod.decide(
            ocr_text="premium status async def env filtered",
            mirage_lines=["premium", "status", "async def", "env", "queue"])
        self.assertEqual(v["decided"], "mirage")

    def test_ocr_picks_when_mirage_adds_nothing(self):
        v = vision_mod.decide(
            ocr_text="hello world", mirage_lines=[])
        self.assertEqual(v["decided"], "ocr")
        v = vision_mod.decide("hello world", ["lorem ipsum dolor"])
        self.assertEqual(v["decided"], "ocr")

    def test_why_names_the_coverage(self):
        v = vision_mod.decide("alpha beta gamma", ["alpha beta gamma delta epsilon"])
        self.assertEqual(v["decided"], "mirage")
        self.assertIn("3/5", v["why"])


class TestHttpDialectRouting(unittest.TestCase):
    """Transport routing must never park a lock behind a client-held stream.

    Live failure this guards (2026-11-24): a client reconnecting after a daemon
    restart sent GET with its old session id. The handler adopted that id,
    created a transport, and awaited the SSE response INSIDE
    `_session_creation_lock`, so the lock was held for the life of the stream.
    Every later POST waited on it and never reached the app (GET still
    answered, which made it look like a partial outage). Reproduced
    deterministically by one GET with an unknown id, then a POST.
    """

    def setUp(self):
        import mcp.server.streamable_http_manager as mcp_sm

        from smartcomp import server as server_mod

        server_mod._install_resilient_session_handler()
        self.sm = mcp_sm
        self.manager = mcp_sm.StreamableHTTPSessionManager(app=unittest.mock.Mock())
        self.manager._task_group = None  # not started yet: the startup window
        self.handler = mcp_sm.StreamableHTTPSessionManager._handle_stateful_request

    def _run(self, method: str, headers: list[tuple[bytes, bytes]], body: bytes = b""):
        scope = {
            "type": "http",
            "method": method,
            "path": "/mcp",
            "headers": headers,
            "query_string": b"",
            "http_version": "1.1",
            "scheme": "http",
            "server": ("127.0.0.1", 8791),
            "client": ("127.0.0.1", 55555),
            "root_path": "",
        }
        sent: list[dict] = []

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message):
            sent.append(message)

        async def go():
            await self.handler(self.manager, scope, receive, send)

        # A hang here is the bug: bound it so a regression fails instead of
        # stalling the whole test run.
        asyncio.run(asyncio.wait_for(go(), 5))
        return sent

    def _status(self, sent: list[dict]) -> int:
        starts = [m for m in sent if m["type"] == "http.response.start"]
        self.assertTrue(starts, "no response was started")
        return starts[0]["status"]

    def test_stale_session_get_is_refused_without_creating_a_transport(self):
        sent = self._run("GET", [(b"mcp-session-id", b"deadbeef" * 4),
                                 (b"accept", b"text/event-stream")])
        self.assertEqual(self._status(sent), 405)
        self.assertEqual(self.manager._server_instances, {})

    def test_sessionless_get_is_405(self):
        sent = self._run("GET", [(b"accept", b"text/event-stream")])
        self.assertEqual(self._status(sent), 405)
        self.assertEqual(self.manager._server_instances, {})

    def test_post_before_lifespan_answers_503_instead_of_stalling(self):
        sent = self._run(
            "POST",
            [(b"content-type", b"application/json"),
             (b"accept", b"application/json, text/event-stream")],
            body=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                             "params": {}}).encode(),
        )
        self.assertEqual(self._status(sent), 503)
        self.assertEqual(self.manager._server_instances, {})


class TestToolSurface(unittest.TestCase):
    """The registered MCP surface: element-addressed, no coordinate habits."""

    def setUp(self):
        mcp = unittest.mock.Mock()
        mcp.tool.return_value = lambda fn: fn  # registration is a no-op here
        self.perception = unittest.mock.Mock()
        # Unlocked host by default; the gate tests below patch it explicitly.
        unlocked = unittest.mock.patch.object(lock_mod, "lock_state", return_value=False)
        unlocked.start()
        self.addCleanup(unlocked.stop)
        self.registry = tools_mod.register(
            mcp, self.perception, unittest.mock.Mock(),
            unittest.mock.Mock(), unittest.mock.Mock(),
            display=unittest.mock.Mock())

    def test_context_menu_is_element_addressed(self):
        self.assertNotIn("x", self.registry["context_menu"].__code__.co_varnames)
        self.assertIn("context_menu", tools_mod._ACTIONS_NEEDING_SCREEN)

    def test_coordinate_tools_are_gone(self):
        for gone in ["mouse_move", "click", "double_click", "mouse_down",
                     "mouse_up", "drag"]:
            self.assertNotIn(gone, self.registry, gone)

    def test_lock_mutation_tools_are_gone(self):
        for gone in ["unlock_screen", "keep_awake", "lid_lock", "wake"]:
            self.assertNotIn(gone, self.registry, gone)
        self.assertIn("lock_state", self.registry)

    def test_lock_state_reports_where_screen_tools_run(self):
        with unittest.mock.patch.object(lock_mod, "lock_state", return_value=True), \
                unittest.mock.patch.object(lock_mod, "host_display_isolated",
                                           return_value=False):
            out = self.registry["lock_state"]()
        self.assertEqual(out["route"], "agent display")
        self.assertEqual(out["auto_session_name"], "agent")
        with unittest.mock.patch.object(lock_mod, "lock_state", return_value=True), \
                unittest.mock.patch.object(lock_mod, "host_display_isolated",
                                           return_value=True):
            isolated = self.registry["lock_state"]()
        self.assertEqual(isolated["route"], "host")

    def test_locked_host_refuses_when_the_fallback_is_off(self):
        blocked = {"error": "host display is locked", "locked": True,
                   "hint": "use session_create"}
        with unittest.mock.patch.object(lock_mod, "blocked_reason",
                                        return_value=blocked), \
                unittest.mock.patch.object(lock_mod, "lock_state", return_value=True), \
                unittest.mock.patch.object(lock_mod, "host_display_isolated",
                                           return_value=False), \
                unittest.mock.patch.object(config, "auto_session_enabled",
                                           return_value=False):
            out = self.registry["screen_state"]()
        self.assertFalse(out["ok"])
        self.assertTrue(out["locked"])
        self.perception.get_changes.assert_not_called()

    def test_locked_host_offers_the_agent_display(self):
        blocked = {"error": "host display is locked", "locked": True}
        manager = unittest.mock.Mock()
        manager.list.return_value = [{"name": "agent", "display": ":1001",
                                      "port": 45999, "alive": True, "healthy": True}]
        manager.auth.return_value = ("http://127.0.0.1:45999/mcp", "tok")
        fake = type("S", (), {"manager": staticmethod(lambda: manager)})()
        routed = {"ok": True, "active": {"title": "Agent desktop"}, "elements": []}
        with unittest.mock.patch.object(lock_mod, "blocked_reason",
                                        return_value=blocked), \
                unittest.mock.patch.object(lock_mod, "lock_state", return_value=True), \
                unittest.mock.patch.object(lock_mod, "host_display_isolated",
                                           return_value=False), \
                unittest.mock.patch.object(tools_mod, "_sessions", lambda: fake), \
                unittest.mock.patch("smartcomp.rpc.call_tool",
                                    return_value=dict(routed)) as call:
            out = self.registry["describe_screen"]()
        self.assertTrue(out["ok"])
        self.assertEqual(out["routed"]["session"], "agent")
        self.assertEqual(out["routed"]["display"], ":1001")
        self.assertIn("host display is locked", out["routed"]["reason"])
        call.assert_called_once()
        self.assertEqual(call.call_args.args[2], "describe_screen")
        self.perception.capturer.grab.assert_not_called()

    def test_fallback_failure_keeps_the_explicit_refusal(self):
        blocked = {"error": "host display is locked", "locked": True}
        manager = unittest.mock.Mock()
        manager.list.return_value = []
        manager.create.return_value = {"ok": False, "error": "no free display"}
        fake = type("S", (), {"manager": staticmethod(lambda: manager)})()
        with unittest.mock.patch.object(lock_mod, "blocked_reason",
                                        return_value=blocked), \
                unittest.mock.patch.object(lock_mod, "lock_state", return_value=True), \
                unittest.mock.patch.object(lock_mod, "host_display_isolated",
                                           return_value=False), \
                unittest.mock.patch.object(tools_mod, "_sessions", lambda: fake):
            out = self.registry["screen_state"]()
        self.assertFalse(out["ok"])
        self.assertIn("no free display", out["route_error"])

    def test_window_management_is_not_gated(self):
        # EWMH/selections work while locked and cannot type into the lock
        # screen, so these stay available on a locked host.
        for name in ("list_windows", "move_window", "close_window",
                     "launch_app", "app_status", "session_create"):
            self.assertNotIn(name, tools_mod._ACTIONS_NEEDING_SCREEN)

    def test_deliberate_and_describing_tools_present(self):
        for present in ["context_menu", "describe_screen", "click_element",
                        "type_text", "scroll", "drag_element", "draw_stroke"]:
            self.assertIn(present, self.registry, present)

    def test_describe_screen_needs_a_usable_screen(self):
        # It reads pixels for the palette; on a locked host it must refuse
        # rather than read the lock screen or wake the display.
        self.assertIn("describe_screen", tools_mod._ACTIONS_NEEDING_SCREEN)
        self.assertIn("context_menu", tools_mod._ACTIONS_NEEDING_SCREEN)
        self.assertIn("draw_stroke", tools_mod._ACTIONS_NEEDING_SCREEN)


class TestLaunchDetachment(unittest.TestCase):
    """launch() must put apps outside the cgroup a daemon restart sweeps.

    Measured live: a Firefox launched by the daemon died at the next
    restart's stop sweep (default KillMode=control-group), no OOM, no
    crash -- start_new_session detaches the session, not the cgroup.
    """

    def test_transient_service_when_a_user_manager_exists(self):
        calls = {"run": [], "popen": 0}

        def fake_run(argv, **kwargs):
            calls["run"].append(argv)
            if argv[0] == "systemd-run":
                return unittest.mock.Mock(returncode=0)
            return unittest.mock.Mock(returncode=0, stdout="4242\n")

        def fail_popen(*args, **kwargs):
            calls["popen"] += 1
            raise AssertionError("Popen must not be used on the systemd path")

        with unittest.mock.patch("shutil.which", return_value="/usr/bin/systemd-run"), \
             unittest.mock.patch.object(input_mod.subprocess, "run", fake_run), \
             unittest.mock.patch.object(input_mod.subprocess, "Popen", fail_popen), \
             unittest.mock.patch.dict(os.environ, {"DISPLAY": ":9"}):
            pid = input_mod.launch("firefox")

        self.assertEqual(pid, 4242)
        self.assertEqual(calls["popen"], 0)
        argv = calls["run"][0]
        self.assertEqual(argv[:3], ["systemd-run", "--user", "--collect"])
        self.assertTrue(argv[3].startswith("--unit=smartcomp-app-firefox-"))
        self.assertIn("--setenv=DISPLAY=:9", argv)
        self.assertEqual(argv[-2:], ["--", "firefox"])

    def test_falls_back_to_a_detached_child_without_systemd_run(self):
        recorded = {}

        def fake_popen(cmd, **kwargs):
            recorded.update(kwargs)
            return unittest.mock.Mock(pid=777)

        with unittest.mock.patch("shutil.which", return_value=None), \
             unittest.mock.patch.object(input_mod.subprocess, "Popen", fake_popen):
            pid = input_mod.launch("mousepad")

        self.assertEqual(pid, 777)
        self.assertTrue(recorded["start_new_session"])

    def test_refused_transient_start_falls_back(self):
        def fake_run(argv, **kwargs):
            return unittest.mock.Mock(returncode=1, stdout="", stderr="no")

        def fake_popen(cmd, **kwargs):
            return unittest.mock.Mock(pid=555)

        with unittest.mock.patch("shutil.which", return_value="/usr/bin/systemd-run"), \
             unittest.mock.patch.object(input_mod.subprocess, "run", fake_run), \
             unittest.mock.patch.object(input_mod.subprocess, "Popen", fake_popen):
            pid = input_mod.launch("thunar")

        self.assertEqual(pid, 555)


class TestSessionPrune(unittest.TestCase):
    """Startup pruning: empty husks go regardless of age, live data stays."""

    def setUp(self):
        import tempfile

        self.dir = Path(tempfile.mkdtemp(prefix="smartcomp-test-"))
        patcher = unittest.mock.patch.object(config, "state_dir", lambda: self.dir)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.mem = memory_mod.Memory(path=self.dir / "mem.db")
        # Memory.__init__ already started the current session dir; a second
        # start_session here would create an empty stray that prune removes.

    def _dir(self, name, files=0, age_s=0.0):
        path = self.mem.sessions_dir / name
        path.mkdir(parents=True, exist_ok=True)
        for i in range(files):
            (path / f"shot-{i}.png").write_bytes(b"png")
        if age_s:
            stamp = time.time() - age_s
            os.utime(path, (stamp, stamp))
        return path

    def test_empty_dir_is_pruned_even_when_fresh(self):
        # end_session deletes the screenshots and the removal refreshes the
        # folder mtime -- age alone would keep the husk for another 24 h.
        husk = self._dir("20260101-000000-000", files=0, age_s=0)
        removed = self.mem.prune_sessions(older_than_hours=24)
        self.assertEqual(removed, 1)
        self.assertFalse(husk.exists())

    def test_filled_fresh_dir_stays(self):
        keep = self._dir("20260101-000100-000", files=2, age_s=0)
        self.mem.prune_sessions(older_than_hours=24)
        self.assertTrue(keep.exists())

    def test_filled_old_dir_is_pruned(self):
        old = self._dir("20251201-000000-000", files=1, age_s=48 * 3600)
        self.mem.prune_sessions(older_than_hours=24)
        self.assertFalse(old.exists())

    def test_the_current_session_dir_stays_even_if_empty(self):
        current = self.mem.sessions_dir / self.mem.session
        self.mem.prune_sessions(older_than_hours=24)
        self.assertTrue(current.exists())


class TestRpcWire(unittest.TestCase):
    """The internal JSON-RPC client the doctor and the lock fallback share."""

    def test_sse_frames_are_parsed(self):
        from smartcomp import rpc

        body = ('event: message\n'
                'data: {"jsonrpc":"2.0","id":1,"result":{"tools":[{"name":"a"}]}}\n\n')
        messages = rpc.parse_messages(body, "text/event-stream")
        self.assertEqual(messages[0]["result"]["tools"][0]["name"], "a")

    def test_plain_json_body_is_parsed(self):
        from smartcomp import rpc

        messages = rpc.parse_messages('{"jsonrpc":"2.0","id":2,"result":{}}', "application/json")
        self.assertEqual(messages[0]["id"], 2)

    def test_garbage_does_not_raise(self):
        from smartcomp import rpc

        self.assertEqual(rpc.parse_messages("<html>502</html>", "text/html"), [])
        self.assertEqual(rpc.parse_messages("data: not-json\n\n", "text/event-stream"), [])

    def test_call_tool_prefers_structured_content(self):
        from smartcomp import rpc

        with unittest.mock.patch.object(rpc, "call", return_value={
                "structuredContent": {"ok": True, "elements": [1]},
                "content": [{"type": "text", "text": '{"ok": true}'}],
                "isError": False}):
            out = rpc.call_tool("http://x/mcp", "t", "screen_state", {})
        self.assertEqual(out["elements"], [1])

    def test_call_tool_falls_back_to_the_text_block(self):
        from smartcomp import rpc

        with unittest.mock.patch.object(rpc, "call", return_value={
                "content": [{"type": "text", "text": '{"ok": true, "count": 3}'}],
                "isError": False}):
            out = rpc.call_tool("http://x/mcp", "t", "list_windows", {})
        self.assertEqual(out["count"], 3)

    def test_is_error_marks_the_result(self):
        from smartcomp import rpc

        with unittest.mock.patch.object(rpc, "call", return_value={
                "content": [{"type": "text", "text": "boom"}], "isError": True}):
            out = rpc.call_tool("http://x/mcp", "t", "click_element", {})
        self.assertFalse(out["ok"])


class TestCliCommands(unittest.TestCase):
    """`smartcomp` starts; stop/restart/status/doctor exist and report clearly."""

    def test_bare_smartcomp_starts_the_daemon(self):
        from smartcomp import daemon, service

        with unittest.mock.patch.object(service, "start", return_value=0) as start:
            self.assertEqual(daemon.main([]), 0)
        start.assert_called_once()

    def test_doctor_exit_code_follows_the_checks(self):
        from smartcomp import daemon, service

        report = {"ok": False, "checks": [
            {"name": "service", "ok": False, "detail": "not running"}]}
        with unittest.mock.patch.object(service, "doctor", return_value=report):
            self.assertEqual(daemon.main(["doctor"]), 1)
        report["ok"] = True
        report["checks"][0]["ok"] = True
        with unittest.mock.patch.object(service, "doctor", return_value=report):
            self.assertEqual(daemon.main(["doctor"]), 0)

    def test_status_reports_stopped_without_raising(self):
        from smartcomp import service

        with unittest.mock.patch.object(service, "running_pid", return_value=None), \
                unittest.mock.patch.object(service, "_probe",
                                           return_value=(False, "refused", None)):
            self.assertEqual(service.status(), 1)

    def test_doctor_warns_instead_of_failing_when_the_lock_is_undetectable(self):
        from smartcomp import service

        with unittest.mock.patch.object(service, "unit_installed", return_value=True), \
                unittest.mock.patch.object(service, "running_pid", return_value=123), \
                unittest.mock.patch.object(service, "_probe", return_value=(True, "ok", 55)), \
                unittest.mock.patch.object(service.shutil, "which", return_value="/usr/bin/x"), \
                unittest.mock.patch("smartcomp.lock.is_locked", return_value=None), \
                unittest.mock.patch("smartcomp.rpc.list_tools", return_value=[{}] * 55), \
                unittest.mock.patch("smartcomp.sessions.manager") as mgr, \
                unittest.mock.patch.object(service, "_orphans", return_value=[]):
            mgr.return_value.list.return_value = []
            report = service.doctor()
        lock_check = next(c for c in report["checks"] if c["name"] == "lock state readable")
        self.assertIsNone(lock_check["ok"])  # WARN, not FAIL
        self.assertTrue(any(c["name"] == "mcp tools/list" and c["ok"] for c in report["checks"]))


class TestProfileCloning(unittest.TestCase):
    """profile_args must create its directory so -profile never says Profile Missing."""

    def test_profile_args_creates_target_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            with unittest.mock.patch.dict(os.environ, {"XDG_RUNTIME_DIR": tmp}):
                ff_args = apps_mod.profile_args("firefox", "firefox")
                cr_args = apps_mod.profile_args("chromium", "chromium")
                self.assertEqual(ff_args[0], "-profile")
                self.assertTrue(Path(ff_args[1]).is_dir())
                self.assertTrue(cr_args[0].startswith("--user-data-dir="))
                self.assertTrue(Path(cr_args[0].split("=", 1)[1]).is_dir())


class TestTaskPlanner(unittest.TestCase):
    """A goal becomes one literal plan; the router only picks a template."""

    def test_url_goal_plans_launch_goto_click(self):
        plan = tasks_mod.parse_goal("open firefox, go to example.com and click Sign in")
        self.assertEqual(plan["template"], "goto")
        tools = [s["tool"] for s in plan["plan"]]
        self.assertEqual(tools[:2], ["launch_app", "goto"])
        self.assertIn("click_label", tools)
        goto = next(s for s in plan["plan"] if s["tool"] == "goto")
        self.assertEqual(goto["args"]["url"], "example.com")
        click = next(s for s in plan["plan"] if s["tool"] == "click_label")
        self.assertEqual(click["args"]["label"], "Sign in")

    def test_app_only_goal_opens_and_reports(self):
        plan = tasks_mod.parse_goal("open firefox")
        self.assertEqual(plan["template"], "open")
        self.assertEqual([s["tool"] for s in plan["plan"]], ["launch_app", "report"])

    def test_explicit_arguments_win_over_parsing(self):
        plan = tasks_mod.parse_goal("do the thing", app="firefox", url="https://x.dev/a")
        goto = next(s for s in plan["plan"] if s["tool"] == "goto")
        self.assertEqual(goto["args"]["url"], "https://x.dev/a")
        self.assertEqual(plan["plan"][0]["args"]["name"], "firefox")

    def test_unparseable_goal_only_reads(self):
        plan = tasks_mod.parse_goal("what is on screen")
        self.assertEqual([s["tool"] for s in plan["plan"]], ["report"])


class TestPolicyRouter(unittest.TestCase):
    """The tiny router is a word-level classifier saved as JSON, not an LLM."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "policy.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_no_data_means_no_prediction(self):
        self.assertIsNone(policy_mod.Router(self.path).predict("open firefox"))

    def test_learns_verb_mapping_and_persists(self):
        r = policy_mod.Router(self.path)
        for _ in range(8):
            r.observe("start firefox now", "open", True)
            r.observe("visit example.com", "goto", True)
        best, prob = r.predict("start firefox")
        self.assertEqual(best, "open")
        self.assertGreater(prob, 0.5)
        again = policy_mod.Router(self.path)
        self.assertEqual(again.predict("start firefox")[0], "open")
        self.assertGreater(again.status()["bytes"], 0)
        self.assertLess(again.status()["bytes"], 200_000)  # stays tiny

    def test_failed_task_does_not_reinforce_its_template(self):
        r = policy_mod.Router(self.path)
        for _ in range(8):
            r.observe("open firefox", "goto", False)
        best, _ = r.predict("open firefox")
        self.assertNotEqual(best, "goto")


class TestTaskQueue(unittest.TestCase):
    """One worker, arrival order, no overlap; results are one small report."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.memory = memory_mod.Memory(path=Path(self.tmp.name) / "m.db")
        self.calls = []
        self.lock = threading.Lock()

        def slow(tag: str = "") -> dict:
            start = time.time()
            time.sleep(0.25)
            with self.lock:
                self.calls.append((tag, start, time.time()))
            return {"ok": True, "tag": tag}

        def list_windows() -> dict:
            return {"windows": [{"id": "0x1", "title": "Example", "active": True}]}

        def read_window(window_id: str, raise_first: bool = True) -> dict:
            return {"ok": True, "lines": [{"x": "Hello"}, {"x": "World"}]}

        self.registry = {"slow": slow, "list_windows": list_windows,
                         "read_window": read_window}
        self.runner = tasks_mod.Runner(self.registry, self.memory,
                                       router=policy_mod.Router(Path(self.tmp.name) / "p.json"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_two_tasks_serialize_in_arrival_order(self):
        out = {}

        def go(tag):
            out[tag] = self.runner.submit(tag, [{"tool": "slow", "args": {"tag": tag}}], "custom")

        a = threading.Thread(target=go, args=("first",))
        a.start()
        time.sleep(0.05)
        b = threading.Thread(target=go, args=("second",))
        b.start()
        a.join(5)
        b.join(5)
        self.assertTrue(out["first"]["ok"] and out["second"]["ok"])
        order = sorted(self.calls, key=lambda c: c[1])
        self.assertEqual([c[0] for c in order], ["first", "second"])
        self.assertGreaterEqual(order[1][1], order[0][2])  # no overlap

    def test_report_step_returns_window_text(self):
        result = self.runner.submit("read it", [{"tool": "report", "args": {}}], "custom")
        self.assertTrue(result["ok"])
        self.assertEqual(result["answer"]["window"], "Example")
        self.assertEqual(result["answer"]["text"], ["Hello", "World"])

    def test_failed_step_stops_the_plan_with_its_own_error(self):
        result = self.runner.submit("x", [{"tool": "slow", "args": {"tag": "a"}},
                                          {"tool": "nope", "args": {}}], "custom")
        self.assertFalse(result["ok"])
        self.assertEqual(result["failed_at"], "nope")
        self.assertIn("unknown step", result["error"])

    def test_successful_task_is_saved_and_replayed(self):
        self.runner.submit("open the example page", [{"tool": "slow", "args": {"tag": "a"}}], "open")
        found = tasks_mod.learned_plan(self.memory, "open the example page")
        self.assertIsNotNone(found)
        name, plan = found
        self.assertEqual(plan[0]["tool"], "slow")
        self.assertEqual(self.memory.skill(name)["description"],
                         "task:open the example page")
        self.assertIsNone(tasks_mod.learned_plan(self.memory, "something else entirely"))


if __name__ == "__main__":
    unittest.main()
