import threading
import queue
import signal
import sys

from sas.runner import Runner
from sas.utils.toml import Config, load_toml
from sas.utils.recorder import FrameRecorder
from sas.utils.model_loader import init_sas_process
from sas.utils.argparser import parse_args
from sas.server_comm import ServerComm


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
            input_queue=self.send_queue,
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
    app = SASApp(raw_config, args)

    if args.profile:
        import cProfile
        print("Running with profiling...")
        cProfile.run("app.run()", "sas_profile.prof")
    else:
        app.run()


if __name__ == "__main__":
    main()
