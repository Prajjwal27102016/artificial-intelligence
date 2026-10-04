import sys
import platform
import time
import os


import cv2
import numpy as np
import mediapipe as mp


from mediapipe.tasks import python
from mediapipe.tasks.python import vision




# ============================================================
# CONFIGURATION
# ============================================================


CAMERA_INDEX = 0


MODEL_PATH = os.path.join(
   os.path.dirname(os.path.abspath(__file__)),
   "hand_landmarker.task"
)


MAX_NUM_HANDS = 2


MIN_HAND_DETECTION_CONFIDENCE = 0.7
MIN_HAND_PRESENCE_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7




# ============================================================
# STARTUP DIAGNOSTICS
# ============================================================


print("1. Python started", flush=True)


print(
   f"2. Python version: {sys.version}",
   flush=True
)


print(
   f"3. MediaPipe version: {mp.__version__}",
   flush=True
)


print(
   f"4. OpenCV version: {cv2.__version__}",
   flush=True
)


print(
   f"5. Model path: {MODEL_PATH}",
   flush=True
)




# ============================================================
# CHECK MODEL FILE
# ============================================================


print(
   "6. Checking Hand Landmarker model...",
   flush=True
)


if not os.path.exists(MODEL_PATH):


   print(
       "\nERROR: hand_landmarker.task was not found.",
       flush=True
   )


   print(
       f"Expected location:\n{MODEL_PATH}",
       flush=True
   )


   print(
       "\nDownload the Hand Landmarker .task model "
       "and place it next to this Python script.",
       flush=True
   )


   sys.exit(1)




print(
   "7. Hand Landmarker model found!",
   flush=True
)




# ============================================================
# CREATE MEDIAPIPE HAND LANDMARKER
# ============================================================


print(
   "8. Creating MediaPipe Hand Landmarker...",
   flush=True
)




BaseOptions = mp.tasks.BaseOptions


HandLandmarker = mp.tasks.vision.HandLandmarker


HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions


RunningMode = mp.tasks.vision.RunningMode




options = HandLandmarkerOptions(


   base_options=BaseOptions(
       model_asset_path=MODEL_PATH
   ),


   running_mode=RunningMode.VIDEO,


   num_hands=MAX_NUM_HANDS,


   min_hand_detection_confidence=
       MIN_HAND_DETECTION_CONFIDENCE,


   min_hand_presence_confidence=
       MIN_HAND_PRESENCE_CONFIDENCE,


   min_tracking_confidence=
       MIN_TRACKING_CONFIDENCE
)




print(
   "9. Options created successfully.",
   flush=True
)




landmarker = HandLandmarker.create_from_options(
   options
)




print(
   "10. MediaPipe Hand Landmarker initialized!",
   flush=True
)




# ============================================================
# GESTURE DETECTION
# ============================================================


print(
   "11. Setting up gesture detection...",
   flush=True
)




def detect_gesture(hand_landmarks):


   landmarks = hand_landmarks


   tip_ids = [4, 8, 12, 16, 20]


   pip_ids = [2, 6, 10, 14, 18]


   extended = 0




   # --------------------------------------------------------
   # Thumb
   # --------------------------------------------------------


   if abs(
       landmarks[tip_ids[0]].x -
       landmarks[pip_ids[0]].x
   ) > 0.04:


       extended += 1




   # --------------------------------------------------------
   # Index, middle, ring, little finger
   # --------------------------------------------------------


   for i in range(1, 5):


       if (
           landmarks[tip_ids[i]].y
           <
           landmarks[pip_ids[i]].y
       ):


           extended += 1




   # --------------------------------------------------------
   # Gesture classification
   # --------------------------------------------------------


   if extended >= 4:


       return "Open"


   elif extended <= 1:


       return "Closed Fist"


   else:


       return "Partial"




print(
   "12. Gesture detection ready.",
   flush=True
)




# ============================================================
# OPEN CAMERA
# ============================================================


print(
   "13. Opening webcam...",
   flush=True
)




if platform.system() == "Windows":


   cap = cv2.VideoCapture(
       CAMERA_INDEX,
       cv2.CAP_DSHOW
   )


else:


   cap = cv2.VideoCapture(
       CAMERA_INDEX
   )




time.sleep(1)




print(
   "14. Checking webcam...",
   flush=True
)




if not cap.isOpened():


   print(
       "ERROR: Could not access the webcam.",
       flush=True
   )


   landmarker.close()


   sys.exit(1)




print(
   "15. Webcam opened successfully!",
   flush=True
)




# ============================================================
# CAMERA PROPERTIES
# ============================================================


width = int(
   cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)


height = int(
   cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)


fps = cap.get(
   cv2.CAP_PROP_FPS
)




print(
   f"16. Camera resolution: {width} x {height}",
   flush=True
)


print(
   f"17. Camera FPS reported: {fps}",
   flush=True
)




# ============================================================
# MAIN LOOP
# ============================================================


print(
   "\nHand Tracking Started! "
   "Press 'q' to quit.",
   flush=True
)




failed_reads = 0


frame_count = 0


timestamp_ms = 0




while True:


   # --------------------------------------------------------
   # READ FRAME
   # --------------------------------------------------------


   success, frame = cap.read()




   if not success:


       failed_reads += 1


       print(
           f"Warning: failed to read frame "
           f"({failed_reads}).",
           flush=True
       )




       if failed_reads > 10:


           print(
               "ERROR: Too many failed frame reads. "
               "Exiting.",
               flush=True
           )


           break




       time.sleep(0.1)


       continue




   failed_reads = 0


   frame_count += 1




   if frame_count <= 5:


       print(
           f"Frame {frame_count} read successfully",
           flush=True
       )




   # --------------------------------------------------------
   # MIRROR IMAGE
   # --------------------------------------------------------


   frame = cv2.flip(
       frame,
       1
   )




   h, w, _ = frame.shape




   # --------------------------------------------------------
   # OPENCV BGR -> MEDIAPIPE RGB
   # --------------------------------------------------------


   frame_rgb = cv2.cvtColor(
       frame,
       cv2.COLOR_BGR2RGB
   )




   # --------------------------------------------------------
   # CREATE MEDIAPIPE IMAGE
   # --------------------------------------------------------


   mp_image = mp.Image(
       image_format=mp.ImageFormat.SRGB,
       data=frame_rgb
   )




   # --------------------------------------------------------
   # TIMESTAMP
   # --------------------------------------------------------


   timestamp_ms = int(
       time.monotonic() * 1000
   )




   # --------------------------------------------------------
   # RUN HAND LANDMARKER
   # --------------------------------------------------------


   if frame_count <= 5:


       print(
           "About to call "
           "landmarker.detect_for_video()...",
           flush=True
       )




   results = landmarker.detect_for_video(
       mp_image,
       timestamp_ms
   )




   if frame_count <= 5:


       print(
           "landmarker.detect_for_video() finished",
           flush=True
       )




   # --------------------------------------------------------
   # DEFAULT GESTURE
   # --------------------------------------------------------


   gesture = "No hand detected"




   # --------------------------------------------------------
   # HAND DETECTION
   # --------------------------------------------------------


   if results.hand_landmarks:


       for idx, hand_landmarks in enumerate(
           results.hand_landmarks
       ):




           # ------------------------------------------------
           # HANDEDNESS
           # ------------------------------------------------


           if (
               results.handedness
               and idx < len(results.handedness)
           ):


               hand_label = (
                   results
                   .handedness[idx][0]
                   .category_name
               )


           else:


               hand_label = "Unknown"




           # ------------------------------------------------
           # GESTURE
           # ------------------------------------------------


           gesture = detect_gesture(
               hand_landmarks
           )




           # ------------------------------------------------
           # DRAW HAND CONNECTIONS
           # ------------------------------------------------


           connections = (
               mp.tasks.vision
               .HandLandmarksConnections
               .HAND_CONNECTIONS
           )




           for connection in connections:


               start_idx = connection.start


               end_idx = connection.end




               start_landmark = (
                   hand_landmarks[start_idx]
               )


               end_landmark = (
                   hand_landmarks[end_idx]
               )




               start_x = int(
                   start_landmark.x * w
               )


               start_y = int(
                   start_landmark.y * h
               )


               end_x = int(
                   end_landmark.x * w
               )


               end_y = int(
                   end_landmark.y * h
               )




               cv2.line(
                   frame,
                   (start_x, start_y),
                   (end_x, end_y),
                   (0, 255, 0),
                   2
               )




           # ------------------------------------------------
           # DRAW ALL 21 LANDMARKS
           # ------------------------------------------------


           for landmark_id, lm in enumerate(
               hand_landmarks
           ):


               x = int(
                   lm.x * w
               )


               y = int(
                   lm.y * h
               )




               cv2.circle(
                   frame,
                   (x, y),
                   5,
                   (255, 0, 255),
                   cv2.FILLED
               )




           # ------------------------------------------------
           # DRAW FINGERTIP IDS
           # ------------------------------------------------


           fingertip_ids = [
               4, 8, 12, 16, 20
           ]




           for tip_id in fingertip_ids:


               lm = hand_landmarks[tip_id]




               x = int(
                   lm.x * w
               )


               y = int(
                   lm.y * h
               )




               cv2.circle(
                   frame,
                   (x, y),
                   10,
                   (255, 0, 255),
                   cv2.FILLED
               )




               cv2.putText(
                   frame,
                   str(tip_id),
                   (x - 5, y - 15),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.5,
                   (255, 255, 0),
                   2
               )




           # ------------------------------------------------
           # WRIST LABEL
           # ------------------------------------------------


           wrist = hand_landmarks[0]




           wrist_x = int(
               wrist.x * w
           )


           wrist_y = int(
               wrist.y * h
           )




           cv2.putText(
               frame,
               f"{hand_label} Hand",
               (
                   wrist_x - 40,
                   wrist_y + 30
               ),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.7,
               (0, 255, 0),
               2
           )




   # ========================================================
   # GESTURE STATUS
   # ========================================================


   if gesture in [
       "Open",
       "Closed Fist"
   ]:


       status_color = (
           0,
           255,
           0
       )


   else:


       status_color = (
           0,
           165,
           255
       )




   cv2.putText(
       frame,
       f"Gesture: {gesture}",
       (10, 30),
       cv2.FONT_HERSHEY_SIMPLEX,
       1,
       status_color,
       2
   )




   # ========================================================
   # DISPLAY
   # ========================================================


   cv2.imshow(
       "Hand Gesture Detection",
       frame
   )




   if frame_count <= 5:


       print(
           "imshow called",
           flush=True
       )




   # ========================================================
   # QUIT
   # ========================================================


   if (
       cv2.waitKey(1) & 0xFF
       == ord('q')
   ):


       print(
           "Q pressed. Exiting...",
           flush=True
       )


       break




# ============================================================
# CLEANUP
# ============================================================


print(
   "Releasing camera...",
   flush=True
)




cap.release()




print(
   "Closing MediaPipe Hand Landmarker...",
   flush=True
)




landmarker.close()




cv2.destroyAllWindows()




print(
   "Program finished.",
   flush=True
)
