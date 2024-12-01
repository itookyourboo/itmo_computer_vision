import argparse
import cv2

from lab1.impl import cv_adaptive_thresholding_mean, our_adaptive_thresholding_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", type=str, required=True,
                    help="path to input image")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])

    # # Оригинал
    # cv2.imshow("Image", image)
    # cv2.waitKey()

    # Конвертируем в черно-белый
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Grayscale", gray)
    # cv2.waitKey()

    # Реализация opencv
    cv_thresh = cv_adaptive_thresholding_mean(gray)
    cv2.imshow("CV Adaptive Thresholding", cv_thresh)
    cv2.waitKey()

    # Собственная реализация
    our_thresh = our_adaptive_thresholding_mean(gray)
    cv2.imshow("Our Adaptive Thresholding", our_thresh)
    cv2.waitKey()


if __name__ == '__main__':
    main()