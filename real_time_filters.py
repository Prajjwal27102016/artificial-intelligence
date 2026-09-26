import sys
import platform
import time

import cv2
import numpy as np

def apply_filter(image, ftype):
    """Apply a filter to the image based on the filter type."""
    img = image.copy()
    if ftype == "red_tint":
        img[:, :, 1] = img[:, :, 0] = 0
    elif ftype == "green_tint":
        img[:, :, 0] = img[:, :, 2] = 0
    elif ftype == "blue_tint":
        img[:, :, 1] = img[:, :, 2] = 0
    elif ftype == "sobel":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=4)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=4)
        magnitude = cv2.magnitude(sx, sy)          # sqrt(sx^2 + sy^2), correct combination
        sob = cv2.convertScaleAbs(magnitude)        # properly scales/clips to uint8
        img = cv2.cvtColor(sob, cv2.COLOR_GRAY2BGR)
    elif ftype == "canny":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        can = cv2.Canny(gray, 1, 20)
        img = cv2.cvtColor(can, cv2.COLOR_GRAY2BGR)
    elif ftype == "cartoon":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 4)
        edges = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 6, 9
        )
        color = cv2.bilateralFilter(image, 9, 300, 300)
        img = cv2.bitwise_and(color, color, mask=edges)
    return img

def main():
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0)

    time.sleep(1)

    if not cap.isOpened():
        print("Cannot open camera")
        sys.exit(1)

    ftype = "original"
    print("Keys: r=Red, g=Green, b=Blue, s=Sobel, c=Canny, t=Cartoon, q=Quit")

    failed_reads = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            failed_reads += 1
            print(f"Warning: can't receive frame ({failed_reads}).")
            if failed_reads > 10:
                print("Too many failed reads. Exiting.")
                break
            time.sleep(0.1)
            continue
        failed_reads = 0

        out = apply_filter(frame, ftype)
        cv2.imshow("Filter", out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            ftype = "red_tint"
        elif key == ord('g'):
            ftype = "green_tint"
        elif key == ord('b'):
            ftype = "blue_tint"
        elif key == ord('s'):
            ftype = "sobel"
        elif key == ord('c'):
            ftype = "canny"
        elif key == ord('t'):
            ftype = "cartoon"
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
