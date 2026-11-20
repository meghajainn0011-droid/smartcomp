"""Unit tests for the parts that do not need a display.

Run with: .venv/bin/python -m unittest discover -s tests -p 'test_*.py'
"""

from __future__ import annotations

import subprocess
import os
import sys
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
from smartcomp import vision as vision_mod  # noqa: E402
from Xlib import XK  # noqa: E402  (keysym ground truth for alias tests)
from smartcomp import awake as awake_mod  # noqa: E402
from smartcomp import input as input_mod  # noqa: E402
from smartcomp import memory as memory_mod  # noqa: E402
from smartcomp.awake import _parse_xset  # noqa: E402
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
            return [TextLine(x=120, y=25, w=800, h=30,
                             text="replaced by held modifier", conf=92.0)]

        with unittest.mock.patch.object(Ocr, "_tesseract_words", side_effect=side_effect):
            out = Ocr().read(crop, deep=deep)
        return out, shapes

    def test_scan_path_rescues_when_main_pass_fails(self):
        out, shapes = self._run(deep=False)
        self.assertEqual([line.text for line in out], ["replaced by held modifier"])
        # strip-local (120, 25) + pad origin (40, 60), scale 1.0
        self.assertEqual((out[0].x, out[0].y), (160, 85))
        self.assertGreaterEqual(len(shapes), 2)  # primary + strip pass ran

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


class TestLockCredential(unittest.TestCase):
    """The unlock credential must be readable only by its owner."""

    def setUp(self):
        import tempfile

        self.dir = tempfile.mkdtemp(prefix="smartcomp-test-")
        patcher = unittest.mock.patch.object(
            lock_mod, "credential_path", lambda: Path(self.dir) / "unlock"
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _write(self, mode: int, text: str = "pw\n") -> Path:
        p = Path(self.dir) / "unlock"
        p.write_text(text)
        p.chmod(mode)
        return p

    def test_owner_only_file_is_accepted(self):
        self._write(0o600)
        self.assertEqual(lock_mod.read_credential(), "pw")

    def test_group_or_world_readable_is_refused(self):
        self._write(0o644)
        self.assertIsNone(lock_mod.read_credential())
        self._write(0o640)
        self.assertIsNone(lock_mod.read_credential())

    def test_missing_or_empty_file_is_refused(self):
        self.assertIsNone(lock_mod.read_credential())
        self._write(0o600, text="   \n")
        self.assertIsNone(lock_mod.read_credential())

    def test_availability_matches_readability(self):
        self.assertFalse(lock_mod.credential_available())
        self._write(0o600)
        self.assertTrue(lock_mod.credential_available())


class TestLidPolicy(unittest.TestCase):
    """The xfpm lid trigger must round-trip: nothing on, originals back."""

    def test_lid_nothing_is_the_documented_enum(self):
        # xfpm-enum-glib.h: DPMS=0, SUSPEND=1, HIBERNATE=2, LOCK_SCREEN=3,
        # NOTHING=4. The stock default is LOCK_SCREEN (3), which is what
        # locked this box on lid close; the unattended policy must set 4.
        from smartcomp.awake import LID_TRIGGER_NOTHING

        self.assertEqual(LID_TRIGGER_NOTHING, 4)

    def test_snapshot_handles_unset_properties(self):
        with unittest.mock.patch.object(
            awake_mod, "_xfconf_lid", lambda prop: None
        ):
            snap = awake_mod._lid_policy_snapshot()
        self.assertEqual(
            snap,
            {
                "/xfce4-power-manager/lid-action-on-ac": None,
                "/xfce4-power-manager/lid-action-on-battery": None,
            },
        )

    def test_set_uses_create_flag(self):
        recorded = {}

        def fake_run(args, **kwargs):
            recorded["args"] = args
            return subprocess.CompletedProcess(args, 0)

        with unittest.mock.patch.object(awake_mod.subprocess, "run", fake_run):
            ok = awake_mod._xfconf_set_lid("/xfce4-power-manager/lid-action-on-ac", "4")
        self.assertTrue(ok)
        self.assertIn("-n", recorded["args"])
        self.assertEqual(recorded["args"][-2:], ["-s", "4"])


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
        registry = tools_mod.register(
            mcp, self.perception, unittest.mock.Mock(), self.wins,
            unittest.mock.Mock(), awake=unittest.mock.Mock())
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


class TestPopupDetection(unittest.TestCase):
    """A popup must surface as a first-class block, never buried in a delta."""

    def setUp(self):
        self.wins = unittest.mock.Mock()
        self.perception = unittest.mock.Mock()
        self.perception.capturer.native_size = (1920, 1080)
        self.perception.elements = {}

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

    def test_empty_screen_no_popup(self):
        self.wins.list.return_value = [
            {"id": "0x1", "comm": "brave", "title": "Big normal page",
             "x": 0, "y": 0, "w": 1200, "h": 1000}]
        self.perception.elements = {}
        self.assertEqual(popups_mod.summarize(popups_mod.detect(
            self.wins, self.perception))["count"], 0)


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


class TestToolSurface(unittest.TestCase):
    """The registered MCP surface: element-addressed, no coordinate habits."""

    def setUp(self):
        mcp = unittest.mock.Mock()
        mcp.tool.return_value = lambda fn: fn  # registration is a no-op here
        self.registry = tools_mod.register(
            mcp, unittest.mock.Mock(), unittest.mock.Mock(),
            unittest.mock.Mock(), unittest.mock.Mock(),
            awake=unittest.mock.Mock())

    def test_context_menu_is_element_addressed(self):
        self.assertNotIn("x", self.registry["context_menu"].__code__.co_varnames)
        self.assertIn("context_menu", tools_mod._ACTIONS_NEEDING_SCREEN)

    def test_coordinate_tools_are_gone(self):
        for gone in ["mouse_move", "click", "double_click", "mouse_down",
                     "mouse_up", "drag"]:
            self.assertNotIn(gone, self.registry, gone)

    def test_deliberate_and_describing_tools_present(self):
        for present in ["context_menu", "describe_screen", "click_element",
                        "type_text", "scroll", "drag_element"]:
            self.assertIn(present, self.registry, present)

    def test_describe_screen_needs_a_usable_screen(self):
        # It reads pixels for the palette; under a lock it must be recovered
        # like every other screen-touching tool, not typed into the lock.
        self.assertIn("describe_screen", tools_mod._ACTIONS_NEEDING_SCREEN)
        self.assertIn("context_menu", tools_mod._ACTIONS_NEEDING_SCREEN)


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


if __name__ == "__main__":
    unittest.main()
