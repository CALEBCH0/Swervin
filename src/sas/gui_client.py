import socket
import struct
import queue
import time
import threading
import numpy as np
import cv2
import json

class Client:
    def __init__(self, server_ip, port=65432,
                frame_width=1024,
                frame_height=1024
                ):
        self.server_address = (server_ip, port)
        self.socket = None
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_channel = 3
        self.frame_queue = queue.Queue(maxsize=10)
        self.label_data_queue = queue.Queue(maxsize=10)
        self.running = True

    def connect(self):
        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Increase socket buffer sizes to handle high data rates
        bufsize = 4 * 1024 * 1024  # 4 MB
        self.conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, bufsize)
        self.conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, bufsize)

        self.conn.connect(self.server_address)
        self.running = True
        print(f"Client: Connected to server at {self.server_address[0]}:{self.server_address[1]}")

    def start_receiving(self):
        receive_thread = threading.Thread(target=self.receive, daemon=True)
        receive_thread.start()

    def cleanup(self):
        if self.conn:
            self.conn.close()
        self.running = False
        print("Client: Connection closed and resources cleaned up.")
    
    def stop(self):
        self.running = False

    def send_command(self, command: str):
        try:
            cmd_msg = command.encode('utf-8')
            self.conn.sendall(cmd_msg)
        except Exception as e:
            print(f"Client: Error sending command - {e}")

    def send_terminate(self):
        self.send_command("TERMINATE")

    def send_start_rec(self):
        self.send_command("START_REC")

    def send_stop_rec(self):
        self.send_command("STOP_REC")
    
    def receive(self, debug=False):
        w, h, c = self.frame_width, self.frame_height, self.frame_channel

        while self.running:
            if True:
                # Receive the length of the image data first (4 bytes for an unsigned int)
                image_size_data = self.conn.recv(4)

                if not image_size_data:
                    break

                # Unpack the image size
                image_size = struct.unpack('!I', image_size_data)[0]
                if debug: print("Client: Image size:", image_size)

                # Now receive the image data based on the received size
                img = b''
                while len(img) < image_size:
                    packet = self.conn.recv(image_size - len(img))
                    if not packet:
                        break
                    img += packet
                if debug: print("Client: Received image data length:", len(img))
                if len(img) != image_size:
                    raise ValueError("Client: Incomplete image data received")
                
                # Try to decode the image as JPEG first, if it fails, treat it as raw data
                try:
                    img_array = cv2.imdecode(np.frombuffer(img, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img_array is None:
                        raise ValueError("Client: Failed to decode JPEG image")
                except: # convert raw format
                    # Convert the received bytes to a numpy array and reshape it
                    img_array = np.frombuffer(img, dtype=np.uint8).reshape((h, w, c))

                # Receive the label data
                json_size_data = self.conn.recv(4)
                if not json_size_data:
                    break

                # Unpack the JSON size
                json_size = struct.unpack('!I', json_size_data)[0]
                if debug: print("Client: Label data size:", json_size)

                # Receive JSON data
                json_data = b''
                while len(json_data) < json_size:
                    packet = self.conn.recv(json_size - len(json_data))
                    if not packet:
                        break
                    json_data += packet

                if len(json_data) != json_size:
                    raise ValueError("Client: Incomplete label data received")

                # Parse JSON
                json_str = json_data.decode('utf-8')
                json_obj = json.loads(json_str)
                if debug: print("Client: Received label data:", json_obj)

                # Put the received frame and label data into the queues
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()  # Drop oldest frame
                    except:
                        pass
                self.frame_queue.put(img_array)

                if self.label_data_queue.full():
                    try:
                        self.label_data_queue.get_nowait()  # Drop oldest label data
                    except:
                        pass
                self.label_data_queue.put(json_obj)

            else:
                # except Exception as e:
                print(f"Error receiving data: {e}")
                self.cleanup()
                break
        self.cleanup()