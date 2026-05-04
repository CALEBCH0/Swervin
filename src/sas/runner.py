import time
import cv2
import numpy as np

class Runner:
    def __init__(self, input_queue, output_queue, send_queue, sas_process, recorder=None, benchmark=None):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.send_queue = send_queue
        self.sas_process = sas_process
        self.recorder = recorder
        self.benchmark = benchmark
        self.running = True

    def run(self):
        bad_cnt = 0
        frame_count = 0
        while self.running:
            if not self.input_queue.empty():
                frame = self.input_queue.get()

                if frame is None:
                    bad_cnt += 1
                    if bad_cnt > 3:
                        break

                img_area = frame.shape[0] * frame.shape[1]
                frame_out = frame
                if self.output_queue.empty():
                    print(f"[Runner] Processing frame {frame_count}, shape={frame.shape}")
                    t0 = time.monotonic()
                    result = self.sas_process(frame, img_area)
                    elapsed = time.monotonic() - t0
                    print(f"[Runner] Frame {frame_count} done")
                    frame_count += 1

                    frame_out, results = result
                    results.fps = 1.0 / elapsed if elapsed > 0 else None
                    mask = None
                    if results.bev is not None:
                        mask = results.bev.bev_mask
                    elif results.seg is not None:
                        mask = results.seg.mask
                    if mask is not None:
                        vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
                        vis[mask > 0] = (0, 200, 0)
                        inset = cv2.resize(vis, (200, 200), interpolation=cv2.INTER_NEAREST)
                        h, w = frame_out.shape[:2]
                        frame_out = frame_out.copy()
                        frame_out[0:200, w - 200:w] = inset

                    self.output_queue.put(result)
                if self.send_queue.empty():
                    self.send_queue.put(frame_out)
            else:
                time.sleep(0.01)

    def stop(self):
        self.running = False
