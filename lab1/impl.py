import cv2
import numpy as np


def cv_adaptive_thresholding_mean(image, max_value=255, block_size = 11, C = 4, inversion = True):
    thresh_type = cv2.THRESH_BINARY
    if inversion:
        thresh_type = cv2.THRESH_BINARY_INV
    # https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3
    # Применяем сам алгоритм:
    # maxValue = 255: то значение которое будет задано пикселю, если превышена граница (threshold)
    # adaptiveMethod = ADAPTIVE_THRESH_MEAN_C: использует метод среднего для порогового определения
    # thresholdType = THRESH_BINARY_INV: инверсия, т.е если порог превышен будет 0, иначе 255
    # blockSize: размер окрестности, в рамках которой будет считаться значение для каждого пикселя
    # С: константа, которая будет вычитаться из нашего посчитанного порога (выводится эмпирически)
    return cv2.adaptiveThreshold(image, max_value,
	cv2.ADAPTIVE_THRESH_MEAN_C, thresh_type, block_size, C)

def our_adaptive_thresholding_mean(image, max_value=255, block_size = 11, C = 4, inversion = True):
    assert block_size > 1 and block_size % 2 == 1
    out = np.zeros_like(image)
    height, width = image.shape

    for h in range(height):
        for w in range(width):
            delta = block_size // 2
            height_start, height_end = max(h - delta, 0), min(h + delta, len(image))
            width_start, width_end = max(w - delta, 0), min(w + delta, len(image[0]))
            region = image[height_start:height_end, width_start:width_end]
            average = np.average(region)

            threshold = average - C
            if (not inversion and image[h][w] >= threshold) or (inversion and image[h][w] < threshold):
                out[h][w] = max_value

    return out




