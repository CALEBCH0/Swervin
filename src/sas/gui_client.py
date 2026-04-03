import socket
import struct
import queue
import time
import threading
import numpy as np

class Client:
    def __init__(self, server_ip, port=00000,
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

        
