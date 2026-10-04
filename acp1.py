import cv2
import numpy as np

cap = cv2.VideoCapture(0)

canvas = None
last_x, last_y = 0, 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    # 1. Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2. Skin color limits
    lower_skin = np.array([0, 20, 70])
    upper_skin = np.array([20, 255, 255])

    # 3. Find skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # 4. Find shapes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Get biggest shape
        hand = max(contours, key=cv2.contourArea)

        if cv2.contourArea(hand) > 3000:
            # Get box around hand
            x, y, w, h = cv2.boundingRect(hand)

            # Fingertip position
            finger_x = x + (w // 2)
            finger_y = y

            # Draw circle on finger
            cv2.circle(frame, (finger_x, finger_y), 8, (0, 0, 255), -1)

            # Draw line
            if last_x != 0 and last_y != 0:
                cv2.line(canvas, (last_x, last_y), (finger_x, finger_y), (0, 255, 0), 5)

            last_x = finger_x
            last_y = finger_y
        else:
            last_x, last_y = 0, 0
    else:
        last_x, last_y = 0, 0

    # Combine video and drawing
    output = cv2.add(frame, canvas)

    cv2.imshow('Hand Tracker', output)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros_like(frame)

cap.release()
cv2.destroyAllWindows()