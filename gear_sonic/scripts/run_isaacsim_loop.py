"""Run the Isaac Sim 6.0.1 to Unitree DDS adapter."""

import argparse
from pathlib import Path
import sys
import traceback

from gear_sonic.utils.isaacsim_simulator.configs import SimLoopConfig


def parse_args(argv=None) -> SimLoopConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd-path", required=True)
    parser.add_argument("--robot-path", default="/World/Robots/g1",
                        help="Robot subtree containing exactly one articulation root")
    parser.add_argument("--base-path", default="/World/Robots/g1/pelvis",
                        help="Pelvis rigid-body prim")
    parser.add_argument("--torso-path", default="/World/Robots/g1/torso_link",
                        help="Torso rigid-body prim")
    parser.add_argument("--physics-scene-path", default="/PhysicsScene",
                        help="PhysicsScene prim")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--interface", default="lo",
                        help="DDS interface; empty lets CycloneDDS choose")
    parser.add_argument("--physics-dt", type=float, default=0.005)
    parser.add_argument("--render-fps", type=float, default=25.0,
                        help="GUI rendering rate; physics and DDS remain at physics-dt")
    parser.add_argument("--render-every", type=int, default=None,
                        help="Legacy override: render once per N physics steps")
    parser.add_argument("--record-video", default=None, metavar="OUTPUT.mp4",
                        help="Capture RGB frames and encode MP4 in a worker process")
    parser.add_argument("--video-fps", type=float, default=20.0)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--video-camera-path", default="/OmniverseKit_Persp")
    parser.add_argument("--command-timeout", type=float, default=0.5)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--inspect", action="store_true",
                        help="Print ten observations, then start the DDS bridge")
    parser.add_argument("--inspect-only", action="store_true",
                        help="Print ten observations and exit without DDS")
    parser.add_argument("--profile", action="store_true",
                        help="Print per-block avg/max timing once per second")
    parser.add_argument("--legacy-state-reader", action="store_false",
                        dest="use_cpp_data_view",
                        help="Use individual Python tensor getters instead of batched C++ views")
    hands = parser.add_mutually_exclusive_group()
    hands.add_argument("--with-hands", dest="with_hands", action="store_true",
                       help="Require integrated seven-DOF Dex3 hands")
    hands.add_argument("--without-hands", dest="with_hands", action="store_false",
                       help="Use only the 29 body joints, even if Dex3 joints are integrated")
    parser.set_defaults(with_hands=None)
    parser.add_argument("--no-realtime", dest="realtime", action="store_false")
    return SimLoopConfig(**vars(parser.parse_args(argv)))


def main(argv=None) -> None:
    config = parse_args(argv)
    vendored_sdk = Path(__file__).resolve().parents[2] / "external_dependencies/unitree_sdk2_python"
    if vendored_sdk.is_dir():
        sys.path.insert(0, str(vendored_sdk))
    # SimulationApp parses sys.argv again and forwards unknown options to Kit.
    # Keep this adapter's CLI flags out of the Kit command line
    sys.argv = sys.argv[:1]
    from isaacsim import SimulationApp

    # DDS is closed explicitly below. Fast Kit shutdown avoids menu callbacks
    # racing a USD context that has already been destroyed during app.close()
    app = SimulationApp({"headless": config.headless, "fast_shutdown": True})
    simulator = None
    exit_code = 0
    try:
        from gear_sonic.utils.isaacsim_simulator.simulator_factory import SimulatorFactory

        simulator = SimulatorFactory.create_simulator(config, app)
        SimulatorFactory.start_simulator(simulator)
    except BaseException:
        exit_code = 1
        traceback.print_exc()
    finally:
        if simulator is not None:
            simulator.close()
        # This is a standalone process and BaseSimulator already stops the
        # timeline, zeros efforts, and closes DDS. Full Kit cleanup can race
        # GUI/RTX worker threads in Isaac Sim 6.0.1 and segfault in
        # SimulationApp.close(); use its documented immediate-exit path.
        app.close(skip_cleanup=True, exit_code=exit_code)


if __name__ == "__main__":
    main()
