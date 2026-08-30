import cv2

image = cv2.imread('happy.png')

cv2.namedWindow('loaded Image', cv2.WINDOW_NORMAL)  

cv2.resizeWindow('loaded Image', 840,600)  

cv2.imshow('loaded Image', image)

cv2.waitKey(0)  

cv2.destroyAllWindows()  

print(f"Image Dimensions: {image.shape}")  # Height, Width, Channels