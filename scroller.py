#scroll up and down


import sys
import platform
import time
import os


import cv2
import pyautogui
import mediapipe as mp


# ============================================================
# CONFIGURATION
# ============================================================


MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")


if not os.path.exists(MODEL_PATH):
   print(f"ERROR: hand_landmarker.task not found at:\n{MODEL_PATH}")
   print("Download it from:")
   print("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
   sys.exit(1)


SCROLL_SPEED = 300
SCROLL_DELAY = 1
CAM_WIDTH, CAM_HEIGHT = 640, 480


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(
   base_options=BaseOptions(model_asset_path=MODEL_PATH),
   running_mode=RunningMode.VIDEO,
   num_hands=1,
   min_hand_detection_confidence=0.7,
   min_tracking_confidence=0.7
)


landmarker = HandLandmarker.create_from_options(options)


mp_draw_connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS


# Landmark indices: tip and its PIP joint (2 below), for index/middle/ring/pinky
TIP_IDS = [8, 12, 16, 20]
THUMB_TIP, THUMB_IP = 4, 3


def detect_gesture(landmarks, handedness):
   fingers = []


   for tip in TIP_IDS:
       if landmarks[tip].y < landmarks[tip - 2].y:
           fingers.append(1)


   thumb_tip = landmarks[THUMB_TIP]
   thumb_ip = landmarks[THUMB_IP]
   if (handedness == "Right" and thumb_tip.x > thumb_ip.x) or (handedness == "Left" and thumb_tip.x < thumb_ip.x):
       fingers.append(1)


   if sum(fingers) == 5:
       return "scroll_up"
   elif len(fingers) == 0:
       return "scroll_down"
   else:
       return "none"


# ============================================================
# CAMERA SETUP
# ============================================================


if platform.system() == "Windows":
   cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
else:
   cap = cv2.VideoCapture(0)


cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)


time.sleep(1)


if not cap.isOpened():
   print("Error: Webcam not accessible.")
   landmarker.close()
   sys.exit(1)


last_scroll = 0
p_time = 0
failed_reads = 0


print("Gesture Scroll Control Active\nOpen palm: Scroll Up\nFist: Scroll Down\nPress 'q' to exit")


while True:
   success, img = cap.read()
   if not success:
       failed_reads += 1
       print(f"Warning: failed to read frame ({failed_reads}).")
       if failed_reads > 10:
           print("Error: Too many failed frame reads. Exiting.")
           break
       time.sleep(0.1)
       continue
   failed_reads = 0


   img = cv2.flip(img, 1)
   img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


   mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
   timestamp_ms = int(time.monotonic() * 1000)
   results = landmarker.detect_for_video(mp_image, timestamp_ms)


   gesture, handedness = "none", "Unknown"


   if results.hand_landmarks:
       for hand, hand_info in zip(results.hand_landmarks, results.handedness):
           handedness = hand_info[0].category_name
           gesture = detect_gesture(hand, handedness)


           h, w = img.shape[:2]
           for connection in mp_draw_connections:
               start = hand[connection.start]
               end = hand[connection.end]
               cv2.line(img,
                        (int(start.x * w), int(start.y * h)),
                        (int(end.x * w), int(end.y * h)),
                        (0, 255, 0), 2)


           if (time.time() - last_scroll) > SCROLL_DELAY:
               if gesture == "scroll_up":
                   pyautogui.scroll(SCROLL_SPEED)
               elif gesture == "scroll_down":
                   pyautogui.scroll(-SCROLL_SPEED)
               last_scroll = time.time()


   fps = 1 / (time.time() - p_time) if (time.time() - p_time) > 0 else 0
   p_time = time.time()


   cv2.putText(img, f"FPS: {int(fps)} | Hand: {handedness} | Gesture: {gesture}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
   cv2.imshow("Gesture Control", img)


   if cv2.waitKey(1) & 0xFF == ord('q'):
       break


cap.release()
landmarker.close()
cv2.destroyAllWindows()
