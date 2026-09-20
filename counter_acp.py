import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('cat.jpg')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_image, cmap='gray')
plt.axis('off')
plt.show()

cropped_image = image[100:300, 200:400]
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

h, w = image.shape[:2]
center = (w // 2, h // 2)

matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated_image = cv2.warpAffine(image, matrix, (w, h))

plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

bright_matrix = np.ones(image.shape, dtype="uint8") * 50
brightened_image = cv2.add(image, bright_matrix)

plt.imshow(cv2.cvtColor(brightened_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

cv2.imwrite('cat_gray.jpg', gray_image)
cv2.imwrite('cat_cropped.jpg', cropped_image)
cv2.imwrite('cat_rotated.jpg', rotated_image)
cv2.imwrite('cat_brightened.jpg', brightened_image)