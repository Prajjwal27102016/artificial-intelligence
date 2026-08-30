import cv2

img = cv2.imread("2happy.png")

small = cv2.resize(img, (300, 200))
medium = cv2.resize(img, (600, 400))
large = cv2.resize(img, (900, 600))

cv2.imwrite("small.png", small)
cv2.imwrite("medium.png", medium)
cv2.imwrite("large.png", large)

cv2.imshow("Small", small)
cv2.imshow("Medium", medium)
cv2.imshow("Large", large)

cv2.waitKey(0)
cv2.destroyAllWindows()