import math
import time
import os
from PIL import Image
import numpy as np
import cv2
import socket
import queue
import threading

class CameraDataGenerator:
    def __init__(self, camera_index=0, width=1920, height=1080, crop=False, flip=False):
        self.width = width
        self.height = height
        self.crop = crop
        # TODO: adjust for webcam
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        if not self.cap.isOpened():
            raise Exception(f"Camera with index {camera_index} could not be opened.")


    def __iter__(self):
        return self
    
    def __next__(self):
        ret, frame = self.cap.read()

        # recording raw camera input
        # cv2.imwrite("./tests/data/camera/camera.png", frame)

        if not ret:
            raise Exception("Failed to read frame from camera.")
        else:
            frame = cv2.flip(frame, 1)

        if not frame.shape[1] == self.width and not frame.shape[0] == self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        return frame  # cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    def release(self):
            self.cap.release()

def sigmoid(image, k=0.001):
    def _sigmoid(x, k=1.0):
        return 1 / (1 + np.exp(-k * x))
    ir_image_sigmoid = _sigmoid(image, k)
    ir_image_sigmoid = cv2.normalize(ir_image_sigmoid, None, 0, 255, cv2.NORM_MINMAX)
    ir_image_sigmoid = np.uint8(ir_image_sigmoid)
    return ir_image_sigmoid

class ImageDataGenerator:
    def __init__(self, directory):
        self.directory = directory
        self.image_files = [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if not self.image_files:
            raise Exception(f"No image files found in directory {directory}.")
        self.index = 0
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.image_files):
            return None

        image_path = os.path.join(self.directory, self.image_files[self.index])
        image = Image.open(image_path)
        # image = np.array(image)
        image_data = np.array(image)
        self.index += 1

        return image_data