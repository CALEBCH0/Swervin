import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Swervin SAS")
    parser.add_argument(
        "--config", type=str, default="config.toml",
        help="Path to config file"
    )
    parser.add_argument(
        "-p", "--profile", action="store_true",
        help="Enable performance profiling"
    )
    parser.add_argument(
        "-b", "--benchmark", action="store_true",
        help="Enable benchmarking mode"
    )
    parser.add_argument(
        "-r", "--record", type=str, default="default",
        help="Session name for recorder"
    )
    parser.add_argument(
        "--calibrate-bev", action="store_true",
        help="Run interactive BEV calibration and save homography, then exit"
    )
    return parser.parse_args()
