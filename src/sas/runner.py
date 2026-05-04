import time
import cv2
import numpy as np


def _build_bev_panel(results, frame_h: int):
    """
    Build an H×H side panel for the BEV visualization.

    Priority: BEV mask (with geometry overlay) > seg mask (no geometry) > None.
    Geometry is drawn in BEV pixel space then scaled to the panel size.
    """
    size = frame_h  # panel is square: frame_h × frame_h

    bev_mask = results.bev.bev_mask if results.bev is not None else None
    seg_mask = results.seg.mask if results.seg is not None else None

    source_mask = bev_mask if bev_mask is not None else seg_mask
    if source_mask is None:
        return None

    vis = np.zeros((*source_mask.shape, 3), dtype=np.uint8)
    vis[source_mask > 0] = (0, 200, 0)
    panel = cv2.resize(vis, (size, size), interpolation=cv2.INTER_NEAREST)

    # Geometry overlay — only available in BEV space
    if bev_mask is not None and results.geometry is not None:
        bev_h, bev_w = bev_mask.shape
        sx = size / bev_w
        sy = size / bev_h
        geom = results.geometry

        # Centerline (yellow)
        pts = geom.centerline  # (N, 2) [x, y]
        for i in range(len(pts) - 1):
            p1 = (int(pts[i][0] * sx),     int(pts[i][1] * sy))
            p2 = (int(pts[i + 1][0] * sx), int(pts[i + 1][1] * sy))
            cv2.line(panel, p1, p2, (0, 255, 255), 2)

        # Lookahead point (red dot)
        lx, ly = geom.lookahead
        cv2.circle(panel, (int(lx * sx), int(ly * sy)), 7, (0, 0, 255), -1)

    return panel


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

                    panel = _build_bev_panel(results, frame_out.shape[0])
                    if panel is not None:
                        frame_out = np.hstack([frame_out, panel])

                    self.output_queue.put(result)
                if self.send_queue.empty():
                    self.send_queue.put(frame_out)
            else:
                time.sleep(0.01)

    def stop(self):
        self.running = False
