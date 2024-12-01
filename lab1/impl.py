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
    # blockSize = 21: размер окрестности, в рамках которой будет считаться значение для каждого пикселя
    # С = 7: константа, которая будет вычитаться из нашего посчитанного порога (выводится эмпирически)
    return cv2.adaptiveThreshold(image, max_value,
	cv2.ADAPTIVE_THRESH_MEAN_C, thresh_type, block_size, C)

def our_adaptive_thresholding_mean(image, max_value=255, block_size = 11, C = 4, inversion = True):
    out = np.zeros_like(image)
    height, width = image.shape
    height = height // block_size + 1
    width = width // block_size + 1

    for h in range(height):
        for w in range(width):
            height_start, height_end = h * block_size, (h + 1) * block_size
            width_start, width_end = w * block_size, (w + 1) * block_size
            region = image[height_start:height_end, width_start:width_end]
            average = np.average(region)

            threshold = average - C
            # applying threshold
            for i in range(height_start, min(height_end, len(image))):
                for j in range(width_start, min(width_end, len(image[0]))):
                    if (not inversion and image[i][j] >= threshold) or (inversion and image[i][j] < threshold):
                        out[i][j] = max_value

    return out




