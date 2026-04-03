import time
import threading
import queue
import signal
import cv2
import numpy as np
import pycuda.driver as cuda
from dataclasses import dataclass
import pstats
import sys

from sas.runner import Runner
from sas.utils.toml import Config, load_toml
from sas.utils.recorder import FrameRecorder
from sas.utils.benchmark import SASBenchmark
from sas.utils.model_loader import init_sas_process

@dataclass
class AppConfig:
    host_ip: str
    frame_width: int
    frame_height: int
    source: str = ['camera', 'image'][0]
    img_dir: str = "./saved/"
    target_fps: int = 37.5
    cam_index: int = 0
    cam_flip: bool = True
    record_interval: int = 1
    record_buffer_size: int = 1000
    profile: bool = "--profile" in sys.argv

class SASApp:
    def __init__(self, app_config, args):
        # Create queues
        self.input_queue = queue.Queue(maxsize=1)
        self.output_queue = queue.Queue(maxsize=1)
        self.send_queue = queue.Queue(maxsize=1)

        # Initialize components
        self.config = app_config
        if app_config.benchmark:
            self.benchmark = SASBenchmark()
        self.recorder = FrameRecorder(
            record_interval=app_config.record_interval,
            buffer_size=app_config.record_buffer_size,
            session_name=args.record
        )

        # Load models 
        toml_config = Config(load_toml(args.config))
        # self.benchmark.start_benchmark("model_loading_time") TODO: pass label and function to measure instead?
        self.sas_process = init_sas_process(toml_config, self.benchmark)
        # self.benchmark.end_benchmark("model_loading_time")

        # Create runner
        self.runner = Runner(
            self.input_queue,
            self.output_queue,
            self.send_queue,
            self.sas_process,
            self.recorder,
            self.benchmark
        )

        # Initialize server communication
        self.comm = ServerComm(
            host_ip=app_config.host_ip,
            input_queue=self.send_queue,
            output_queue=self.output_queue,
            send_queue=self.send_queue,
            frame_width=app_config.frame_width,
            frame_height=app_config.frame_height,
            source=app_config.source,
            img_dir=app_config.img_dir,
            target_fps=app_config.target_fps,
            cam_index=app_config.cam_index,
            cam_flip=app_config.cam_flip,
            runner=self.runner,
            recorder=self.recorder
        )

        # Setup signal handler
        signal.signal(signal.SIGTERM, self.comm.signal_handler)
        signal.signal(signal.SIGINT, self.comm.signal_handler)

    def run(self):
        """Main exdecution loop for the SAS application."""
        self.comm.setup_socket()
        self.comm.accept_connection()

        comm_thread = threading.Thread(target=self.comm.run)
        comm_thread.start()

        try:
            self.runner.run()
        finally:
            # save_benchmark()
            if hasattr(self.comm, 'shutdown_event'):
                self.comm.shutdown_event.set()
                comm_thread.join(timeout=2.0)
            print("App Shutdown complete.")

def main():
    args = parse_args()
    app_config = AppConfig()
    app = SASApp(app_config, args)

    if args.profile:
        import cProfile
        print("Running with profiling...")
        cProfile.run('app.run()', 'sas_profile.prof')
    else:
        app.run()

if __name__ == "__main__":
    main()