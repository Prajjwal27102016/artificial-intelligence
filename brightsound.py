import sys
import platform
import time
import os


import cv2
import numpy as np
import mediapipe as mp
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as sbc


# ============================================================
# CONFIGURATION
# ============================================================


MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")


if not os.path.exists(MODEL_PATH):
   print(f"ERROR: hand_landmarker.task not found at:\n{MODEL_PATH}")
   print("Download it from:")
   print("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
   sys.exit(1)


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(
   base_options=BaseOptions(model_asset_path=MODEL_PATH),
   running_mode=RunningMode.VIDEO,
   num_hands=2,
   min_hand_detection_confidence=0.7,
   min_hand_presence_confidence=0.7,
   min_tracking_confidence=0.7
)


landmarker = HandLandmarker.create_from_options(options)


# Landmark indices (same numbering as legacy API)
TH, IX = 4, 8  # THUMB_TIP, INDEX_FINGER_TIP


# ============================================================
# AUDIO SETUP (pycaw)
# ============================================================


try:
   dev = AudioUtilities.GetDefaultOutputDevice() if hasattr(AudioUtilities, "GetDefaultOutputDevice") else AudioUtilities.GetSpeakers()
   volctl = dev.EndpointVolume.QueryInterface(IAudioEndpointVolume)
   minv, maxv = volctl.GetVolumeRange()[:2]
except Exception as e:
   print(f"Pycaw error: {e}")
   landmarker.close()
   sys.exit(1)


# ============================================================
# CAMERA SETUP
# ============================================================


if platform.system() == "Windows":
   cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
else:
   cap = cv2.VideoCapture(0)


time.sleep(1)


if not cap.isOpened():
   print("Error: Webcam not accessible.")
   landmarker.close()
   sys.exit(1)


WIN = "Hand Gesture Control"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)


failed_reads = 0


while True:
   ok, img = cap.read()
   if not ok:
       failed_reads += 1
       print(f"Warning: failed to read frame ({failed_reads}).")
       if failed_reads > 10:
           print("Error: Too many failed frame reads. Exiting.")
           break
       time.sleep(0.1)
       continue
   failed_reads = 0


   img = cv2.flip(img, 1)
   h, w = img.shape[:2]


   img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
   timestamp_ms = int(time.monotonic() * 1000)


   res = landmarker.detect_for_video(mp_image, timestamp_ms)


   if res.hand_landmarks:
       for i, hand in enumerate(res.hand_landmarks):
           if res.handedness and i < len(res.handedness):
               label = res.handedness[i][0].category_name
           else:
               label = "Unknown"


           # Draw all 21 landmarks
           for lm in hand:
               x, y = int(lm.x * w), int(lm.y * h)
               cv2.circle(img, (x, y), 5, (255, 0, 255), cv2.FILLED)


           tp = (int(hand[TH].x * w), int(hand[TH].y * h))
           ip = (int(hand[IX].x * w), int(hand[IX].y * h))
           cv2.circle(img, tp, 10, (255, 0, 0), cv2.FILLED)
           cv2.circle(img, ip, 10, (255, 0, 0), cv2.FILLED)
           cv2.line(img, tp, ip, (0, 255, 0), 3)
           dist = float(np.hypot(ip[0] - tp[0], ip[1] - tp[1]))


           if label == "Left":  # real RIGHT hand -> volume (frame is flipped)
               v = np.interp(dist, [20, 150], [minv, maxv])
               try:
                   volctl.SetMasterVolumeLevel(v, None)
               except Exception as e:
                   print(f"Volume error: {e}")
               bar = int(np.interp(dist, [20, 150], [400, 150]))
               pct = int(np.interp(dist, [20, 150], [0, 100]))
               cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 2)
               cv2.rectangle(img, (50, bar), (85, 400), (255, 0, 0), cv2.FILLED)
               cv2.putText(img, f"{pct}%", (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)


           elif label == "Right":  # real LEFT hand -> brightness
               b = int(np.interp(dist, [20, 150], [0, 100]))
               try:
                   sbc.set_brightness(b)
               except Exception as e:
                   print(f"Brightness error: {e}")
               bar = int(np.interp(dist, [20, 150], [400, 150]))
               x1, x2 = w - 85, w - 50
               cv2.rectangle(img, (x1, 150), (x2, 400), (0, 255, 0), 2)
               cv2.rectangle(img, (x1, bar), (x2, 400), (0, 255, 0), cv2.FILLED)
               cv2.putText(img, f"{b}%", (w - 110, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)


   cv2.imshow(WIN, img)
   k = cv2.waitKey(1) & 0xFF
   if k in (27, ord("q")):
       break
   try:
       if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
           break
   except cv2.error:
       break


cap.release()
landmarker.close()
cv2.destroyAllWindows()
