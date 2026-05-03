import cv2
import numpy as np

def imagenet_preprocess(input, size):
    h, w = size  # size is (height, width)
    input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    input = cv2.resize(input, (w, h), interpolation=cv2.INTER_LINEAR)
    input = input.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    input -= mean[np.newaxis, np.newaxis, :]
    input /= std[np.newaxis, np.newaxis, :]

    return input.transpose(2, 0, 1)

def yolox_preprocess(img, input_size=(320, 320), swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones(
            (input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img, (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR).astype(np.uint8)

    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r