import glob
import os
import threading
import queue
import signal
import sys

import cv2
import numpy as np

from sas.runner import Runner
from sas.utils.toml import Config, load_toml
from sas.utils.recorder import FrameRecorder
from sas.utils.model_loader import init_sas_process
from sas.utils.argparser import parse_args
from sas.server_comm import ServerComm
from sas.utils.bev_transformer import get_src, get_dst, compute_bev_transform


def run_bev_calibration(raw_config):
    bev_cfg = raw_config['bev']
    output_size = tuple(bev_cfg['output_size'])
    save_path = bev_cfg['homography_path']

    img_path = bev_cfg.get('calibration_img')
    if not img_path:
        img_dir = raw_config['app']['img_dir']
        frames = sorted(
            f for ext in ('*.jpg', '*.jpeg', '*.png')
            for f in glob.glob(os.path.join(img_dir, ext))
        )
        if not frames:
            raise FileNotFoundError(f"No images found in {img_dir}")
        img_path = frames[0]

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read calibration image: {img_path}")
    mask_size = tuple(bev_cfg['mask_size'])  # (width, height) matching seg model output
    img = cv2.resize(img, mask_size)

    src = get_src(img)
    if len(src) < 4:
        print("BEV calibration cancelled.")
        return
    dst = get_dst(output_size)
    M, _ = compute_bev_transform(src, dst)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, M)
    print(f"BEV calibration saved to {save_path}. Re-run without --calibrate-bev to start.")


class SASApp:
    def __init__(self, raw_config, args):
        app = raw_config["app"]
        env = raw_config["env"][app["active_env"]]

        self.input_queue = queue.Queue(maxsize=1)
        self.output_queue = queue.Queue(maxsize=1)
        self.send_queue = queue.Queue(maxsize=1)

        self.recorder = FrameRecorder(
            record_interval=app["record_interval"],
            buffer_size=app["record_buffer_size"],
            session_name=args.record
        )

        toml_config = Config(raw_config)
        self.sas_process = init_sas_process(toml_config)

        self.runner = Runner(
            self.input_queue,
            self.output_queue,
            self.send_queue,
            self.sas_process,
            self.recorder,
            benchmark=None
        )

        self.comm = ServerComm(
            host_ip=env["host_ip"],
            input_queue=self.input_queue,
            output_queue=self.output_queue,
            send_queue=self.send_queue,
            frame_width=env["frame_width"],
            frame_height=env["frame_height"],
            source=app["source"],
            img_dir=app["img_dir"],
            target_fps=app["target_fps"],
            cam_index=app["cam_index"],
            cam_flip=app["cam_flip"],
            runner=self.runner,
            recorder=self.recorder
        )

        signal.signal(signal.SIGTERM, self.comm.signal_handler)
        signal.signal(signal.SIGINT, self.comm.signal_handler)

    def run(self):
        self.comm.setup_socket()
        self.comm.accept_connection()

        comm_thread = threading.Thread(target=self.comm.run)
        comm_thread.start()

        try:
            self.runner.run()
        finally:
            if hasattr(self.comm, "shutdown_event"):
                self.comm.shutdown_event.set()
                comm_thread.join(timeout=2.0)
            print("App shutdown complete.")


def main():
    args = parse_args()
    raw_config = load_toml(args.config)

    if args.calibrate_bev:
        run_bev_calibration(raw_config)
        return

    app = SASApp(raw_config, args)

    if args.profile:
        import cProfile
        print("Running with profiling...")
        cProfile.run("app.run()", "sas_profile.prof")
    else:
        app.run()


if __name__ == "__main__":
    main()
