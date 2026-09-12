import cv2
import matplotlib.pyplot as plt

image_path = 'hill.webp'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

height, width, _ = image_rgb.shape

margin = 40
arrow_y = height // 2

start_point = (margin, arrow_y)
end_point = (width - margin, arrow_y)

color = (255, 0, 0)
thickness = 4
tip_length = 0.04

cv2.arrowedLine(image_rgb, start_point, end_point, color, thickness, tipLength=tip_length)
cv2.arrowedLine(image_rgb, end_point, start_point, color, thickness, tipLength=tip_length)

font = cv2.FONT_HERSHEY_SIMPLEX
label = f"Width: {width}px"
font_scale = 1.0
text_thickness = 2

(text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
text_x = (width - text_width) // 2
text_y = arrow_y - 15

cv2.putText(image_rgb, label, (text_x, text_y), font, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)

plt.figure(figsize=(12, 8))
plt.imshow(image_rgb)
plt.title('Image Width Visualization with Bi-directional Arrow')
plt.axis('off')
plt.show()