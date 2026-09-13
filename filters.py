import cv2
import numpy as np

def apply_color_filter(image, filter_type):
    """Apply the specified color filter to the image.

    Note: OpenCV loads images in BGR order, so:
      channel 0 = Blue, channel 1 = Green, channel 2 = Red
    """
    filtered_image = image.copy()

    if filter_type == "red_tint":
        # Keep only Red -> zero out Blue(0) and Green(1)
        filtered_image[:, :, 0] = 0
        filtered_image[:, :, 1] = 0
    elif filter_type == "blue_tint":
        # Keep only Blue -> zero out Green(1) and Red(2)
        filtered_image[:, :, 1] = 0
        filtered_image[:, :, 2] = 0
    elif filter_type == "green_tint":
        # Keep only Green -> zero out Blue(0) and Red(2)
        filtered_image[:, :, 0] = 0
        filtered_image[:, :, 2] = 0
    elif filter_type == "increase_red":
        # Increase intensity of Red channel (index 2), clipped at 255
        filtered_image[:, :, 2] = cv2.add(filtered_image[:, :, 2], 50)
    elif filter_type == "decrease_blue":
        # Decrease intensity of Blue channel (index 0), clipped at 0
        filtered_image[:, :, 0] = cv2.subtract(filtered_image[:, :, 0], 50)
    # "original" or any unrecognized value falls through unchanged

    return filtered_image


def main():
    image_path = 'example2.jpg'  # Provide your image path
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Image not found!")
        return

    filter_type = "original"  # Default filter type

    print("Press the following keys to apply filters:")
    print("r - Red Tint")
    print("b - Blue Tint")
    print("g - Green Tint")
    print("i - Increase Red Intensity")
    print("d - Decrease Blue Intensity")
    print("q - Quit")

    cv2.namedWindow("Filtered Image", cv2.WINDOW_NORMAL)

    try:
        while True:
            filtered_image = apply_color_filter(image, filter_type)
            cv2.imshow("Filtered Image", filtered_image)

            key = cv2.waitKey(0) & 0xFF

            if key == ord('r'):
                filter_type = "red_tint"
            elif key == ord('b'):
                filter_type = "blue_tint"
            elif key == ord('g'):
                filter_type = "green_tint"
            elif key == ord('i'):
                filter_type = "increase_red"
            elif key == ord('d'):
                filter_type = "decrease_blue"
            elif key == ord('q'):
                print("Exiting...")
                break
            else:
                print("Invalid key! Please use 'r', 'b', 'g', 'i', 'd', or 'q'.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

