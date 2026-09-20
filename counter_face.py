import sys
import platform
import time


import cv2


# Load pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


if face_cascade.empty():
   print("Error: Could not load Haar Cascade file.")
   print("If you're on OpenCV 5.0+, make sure you installed 'opencv-contrib-python'")
   print("instead of plain 'opencv-python' — CascadeClassifier moved to contrib in 5.0.")
   sys.exit(1)


# Initialize video capture (use webcam)
if platform.system() == "Windows":
   cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
else:
   cap = cv2.VideoCapture(0)


time.sleep(1)  # let the camera warm up before first read


if not cap.isOpened():
   print("Error: Could not open webcam.")
   sys.exit(1)


failed_reads = 0


while True:
   # Capture frame-by-frame
   ret, frame = cap.read()


   if not ret:
       failed_reads += 1
       print(f"Warning: failed to capture image ({failed_reads}).")
       if failed_reads > 10:
           print("Error: Too many failed frame reads. Exiting.")
           break
       time.sleep(0.1)
       continue


   failed_reads = 0


   # Convert frame to grayscale
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


   # Detect faces in the grayscale image
   faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))


   # Draw rectangles around faces
   for (x, y, w, h) in faces:
       cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 500)


   # Display the count of faces
   font = cv2.FONT_HERSHEY_SIMPLEX
   cv2.putText(frame, f'People Count: {len(faces)}', (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)


   # Display the frame with face detection and people count
   cv2.imshow('Face Tracking and Counting', frame)


   # Exit the loop when the 'q' key is pressed
   if cv2.waitKey(1) & 0xFF == ord('q'):
       break


# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
