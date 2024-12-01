import argparse
import time
from typing import Callable, TypeVar

import cv2

from lab1.impl import cv_adaptive_thresholding_mean, our_adaptive_thresholding_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", type=str, required=True,
                    help="path to input image")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])

    # Оригинал
    cv2.imshow("Image", image)
    cv2.waitKey()

    # Конвертируем в черно-белый
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale", gray)
    cv2.waitKey()

    # Реализация opencv
    time_cv_thresh, cv_thresh = time_elapsed(cv_adaptive_thresholding_mean, gray)
    print(f'Time elapsed on opencv implementation: {time_cv_thresh}')
    cv2.imshow("CV Adaptive Thresholding", cv_thresh)
    cv2.waitKey()

    # Собственная реализация
    time_our_thresh, our_thresh = time_elapsed(our_adaptive_thresholding_mean, gray)
    print(f'Time elapsed on our implementation: {time_our_thresh}')
    cv2.imshow("Our Adaptive Thresholding", our_thresh)
    cv2.waitKey()

T = TypeVar('T')
def time_elapsed(f: Callable[[...], T], *args, **kwargs) -> tuple[float, T]:
    start = time.time()
    res = f(*args, **kwargs)
    end = time.time()
    return end - start, res

if __name__ == '__main__':
    main()