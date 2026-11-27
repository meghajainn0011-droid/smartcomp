"""Command line entry points for the SmartComp perception daemon."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

from . import config
from . import __version__


def _once(args) -> int:
    from .perception import Perception

    p = Perception(display=args.display)
    payload = p.get_changes(force_full=True)
    print(json.dumps(payload, separators=(",", ":")))
    if args.settle:
        time.sleep(args.settle)
        print(json.dumps(p.get_changes(), separators=(",", ":")))
    return 0


def _bench(args) -> int:
    """Measure the pipeline against the plan's budgets (T3/T4/T5).

    Two cases matter and they differ by an order of magnitude: a full screen
    (window switch, first call) and the steady state where nothing moved. Both
    are reported; a single median across them hides the expensive one.
    """
    import os

    import numpy as np

    from .perception import Perception

    p = Perception(display=args.display)
    full_rows: list[dict] = []
    payload_sizes: list[int] = []
    for i in range(args.rounds):
        payload = p.get_changes(force_full=True)
        full_rows.append(payload["stats"])
        payload_sizes.append(len(json.dumps(payload, separators=(",", ":"))))
        time.sleep(args.interval)
    idle = p.get_changes(force_full=False)["stats"]

    def line(name, values, budget_ms=None):
        if not values:
            return
        med = statistics.median(values)
        p95 = float(np.percentile(values, 95))
        verdict = ""
        if budget_ms is not None:
            verdict = "  PASS" if med < budget_ms else f"  FAIL (>{budget_ms}ms)"
        print(f"  {name:<9} median={med:7.1f}ms  p95={p95:7.1f}ms{verdict}")

    load = os.getloadavg()[0]
    print(
        f"rounds={args.rounds} interval={args.interval}s engine={p.ocr.engine} "
        f"cores={os.cpu_count()} load1={load:.2f}"
    )
    print("full screen (T3 capture / T4 OCR / T5 detector, budget 300ms each):")
    line("capture", [r["cap"] for r in full_rows], 300)
    line("diff", [r["diff"] for r in full_rows], 300)
    line("ocr", [r["ocr"] for r in full_rows], 300)
    line("detector", [r["det"] for r in full_rows], 300)
    line("total", [r["total"] for r in full_rows], 300)
    print(f"  payload   median={statistics.median(payload_sizes):.0f} bytes")
    print(f"idle screen: total={idle['total']:.1f}ms tiles={idle['tiles']} (cost of a no-op poll)")
    print(f"elements   {len(p.elements)}")
    if not p.detector.available:
        print(f"detector   UNAVAILABLE: {p.detector.error}")
    sample = [e["x"] for e in p.get_changes(force_full=True).get("add", []) if e["k"] == "t"]
    print("sample text:", sample[:10])
    return 0


def _serve(args) -> int:
    from .server import serve

    serve()
    return 0


def _tray(args) -> int:
    from .tray import main as tray_main

    return tray_main([])


def _start(args) -> int:
    from . import service

    return service.start()


def _stop(args) -> int:
    from . import service

    return service.stop()


def _restart(args) -> int:
    from . import service

    return service.restart()


def _status(args) -> int:
    from . import service

    return service.status()


def _doctor(args) -> int:
    from . import service

    return service.print_doctor(service.doctor())


def _state(args) -> int:
    from .perception import Perception

    p = Perception(display=args.display)
    p.get_changes(force_full=True)
    print(json.dumps(p.state(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--display", default=None, help="X display, e.g. :0.0")
    parser = argparse.ArgumentParser(
        prog="smartcomp",
        description="SmartComp desktop-control daemon. No command starts it.",
        epilog="A locked host keeps working: screen tools run on the agent's "
               "own display without unlocking, waking or inhibiting anything.",
    )
    parser.add_argument("--version", action="version", version=f"smartcomp {__version__}")
    sub = parser.add_subparsers(dest="cmd")

    p_start = sub.add_parser("start", help="start the daemon (the default)", parents=[common])
    p_start.set_defaults(func=_start)
    p_stop = sub.add_parser("stop", help="stop the daemon", parents=[common])
    p_stop.set_defaults(func=_stop)
    p_restart = sub.add_parser("restart", help="restart the daemon", parents=[common])
    p_restart.set_defaults(func=_restart)
    p_status = sub.add_parser("status", help="is it running, and where", parents=[common])
    p_status.set_defaults(func=_status)
    p_doctor = sub.add_parser("doctor", parents=[common],
                              help="run health checks and explain failures")
    p_doctor.set_defaults(func=_doctor)

    p_once = sub.add_parser("once", help="one perception pass as JSON", parents=[common])
    p_once.add_argument("--settle", type=float, default=0.0, help="seconds, then a second pass")
    p_once.set_defaults(func=_once)

    p_bench = sub.add_parser(
        "bench", help="time the pipeline against the plan budgets", parents=[common]
    )
    p_bench.add_argument("--rounds", type=int, default=5)
    p_bench.add_argument("--interval", type=float, default=1.5)
    p_bench.set_defaults(func=_bench)

    sub.add_parser("serve", help="run the MCP HTTP server in the foreground", parents=[common]
                   ).set_defaults(func=_serve)
    sub.add_parser("tray", help="run the panel tray icon", parents=[common]).set_defaults(
        func=_tray
    )
    sub.add_parser("state", help="dump the current element set", parents=[common]).set_defaults(
        func=_state
    )

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-") and argv[0] not in ("--help", "-h", "--version"):
        argv.insert(0, "start")
    args = parser.parse_args(argv)
    if args.display:
        import os

        os.environ["DISPLAY"] = args.display
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
