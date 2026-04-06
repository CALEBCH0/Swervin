import socket
import struct
import numpy as np
import sys
import time
import threading
import queue
import select

class ServerComm:
    def __init__(self, host_ip, input_queue, output_queue, send_queue,
                    img_dir=None,
                    source='image',
                    port=65432,
                    frame_width=640,
                    frame_height=480,
                    target_fps=30,
                    cam_index=0,
                    cam_flip=False,
                    runner=None,
                    recorder=None):
        self.server_address = (host_ip, port)
        self.img_dir = img_dir
        self.sock = None
        self.conn = None
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.send_queue = send_queue
        self._target_fps = target_fps
        self._frame_duration = 1.0 / target_fps
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.runner = runner
        self.recorder = recorder
        self.shutdown_event = threading.Event()

        if source == 'camera':
            from sas.datagenerator import CameraDataGenerator
            self.image_generator = CameraDataGenerator(cam_index, cam_flip, frame_width, frame_height)
        elif source == 'image' and img_dir:
            from sas.datagenerator import ImageDataGenerator
            self.image_generator = ImageDataGenerator(self.img_dir)
        else:
            raise ValueError(f"Unsupported source type: {source}. Must be 'camera' or 'image' with a valid image directory.")

        # Create a queue for buggering frames from image generator
        self.frame_buffer = queue.Queue(maxsize=2)
        self.generator_thread = None
        self.running = False

    def setup_socket(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # for immediate reconnection
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True) # disable Nagle's algorithm
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2**20)
        self.socket.bind(self.server_address)
        self.socket.listen(1)
        print(f"Server: Listening on {self.server_address[0]}:{self.server_address[1]}")
    
    def accept_connection(self):
        self.conn, addr = self.socket.accept()
        print(f"Server: Connection accepted from {addr}")
        self.conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)
        self.conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2**20)

    def send_data(self, image, label_data):
        # print("Server: Sending data", self.frame_height, self.frame_width)
        assert image.dtype == np.uint8, "Image must be in RGB888 format (uint8)"
        # print("Server: Sending data", len(label_data))
        # Pack the image data
        image_data = image.tobytes()
        packed_data = struct.pack('!I', len(image_data)) + image_data + label_data
        self.conn.sendall(packed_data)

    def _frame_generator_thread(self):
        """Thread function to continuously read frames from image generator"""
        while self.running:
            try:
                image = next(self.image_generator)
                if image is not None:
                    # Only put frame if queue has space, otherwise drop oldest
                    if self.frame_buffer.full():
                        try:
                            self.frame_buffer.get_nowait()  # Drop oldest frame
                        except queue.Empty:
                            pass
                    self.frame_buffer.put(image)
            except StopIteration:
                # Handle end of image sequence for ImageDataGenerator
                break
            except Exception as e:
                print(f"Server: Error in frame generator thread: {e}")
                break
            time.sleep(0.01)  # Small sleep to prevent tight loop if generator is fast
    
    def handle_command(self, command_data: bytes):
        """Handle commands received from client"""
        try:
            command = command_data.decode('utf-8')
            # print(f"Server: Received command: {command}")

            if command == 'TERMINATE':
                print("Server: Received TERMINATE command. Shutting down server.")
                self.initiate_shutdown()
            elif command == 'START_REC':
                print("Server: Received START_REC command. Starting recording.")
                if self.recorder:
                    self.recorder.start_recording()
            elif command == 'STOP_REC':
                print("Server: Received STOP_REC command. Stopping recording.")
                if self.recorder:
                    self.recorder.stop_recording()
            else:
                print(f"Server: Unknown command received: {command}")
        except UnicodeDecodeError:
            print("Server: Failed to decode command data. Invalid UTF-8 sequence.")
        except Exception as e:
            print(f"Server: Error handling command: {e}")
    
    def run(self):
        # Start the frame generator thread
        self.running = True
        self.generator_thread = threading.Thread(target=self._frame_generator_thread, daemon=True)
        self.generator_thread.start()

        try:
            while self.running:
                # Check for incoming commands from client
                try:
                    ready = select.select([self.conn], [], [], 0.001) # 1ms timeout
                    if ready[0]:
                        ctrl_data = self.conn.recv(1024) # Adjust buffer size as needed
                        if ctrl_data:
                            self.handle_command(ctrl_data)
                        else:
                            print("Server: Client disconnected.")
                            self.running = False
                            break
                except (socket.error, ConnectionResetError):
                    print("Server: Client connection lost.")
                    self.running = False
                    break
                
                # Get frame from buffer instead of directly from generator
                try:
                    image = self.frame_buffer.get(timeout=0.1) # Wait for frame to be available
                except queue.Empty:
                    time.sleep(0.01)  # No frame available, wait a bit and check again
                    continue
                
                if image is not None:
                    # image = cv2.flip(image, 1)
                    if self.input_queue.empty():
                        self.input_queue.put(image)
                    if not self.output_queue.empty() and not self.send_queue.full():
                        label_values = self.output_queue.get()
                        img_send = self.send_queue.get()
                        self.send_data(img_send, label_values)
                time.sleep(0.01)  # Sleep to maintain target FPS
        finally:
            self.cleanup()

    def signal_handler(self, sig, frame):
        print("Server: Received signal:", sig)
        self.initiate_shutdown()

    def initiate_shutdown(self):
        print("Server: Initiating shutdown...")
        self.stop()
        self.shutdown_event.set() # Notify any waiting threads to exit

        if self.recorder and self.recorder.is_recording():
            self.recorder.stop_recording()

        if self.runner:
            self.runner.stop()

    def stop(self):
        self.running = False

    def cleanup(self):
        # Stop the generator thread
        print("Server: Cleaning up resources...")
        self.running = False
        if self.generator_thread and self.generator_thread.is_alive():
            self.generator_thread.join(timeout=1.0)

        if self.conn:
            self.conn.close()
        if self.socket:
            self.socket.close()
        
        print("Server: Cleanup complete.")