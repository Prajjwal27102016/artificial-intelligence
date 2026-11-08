# ---------- SETUP (given to you) ----------
import cv2
import numpy as np

photo = np.full((120, 100, 3), 180, np.uint8)             # a grey classroom wall
cv2.rectangle(photo, (30, 60), (70, 120), (40, 90, 220), -1)  # the student's shirt
cv2.circle(photo, (50, 40), 18, (150, 180, 220), -1)          # the student's face
cv2.rectangle(photo, (80, 10), (95, 30), (60, 60, 60), -1)    # a poster of a person on the wall

person = np.zeros((120, 100), np.float32)                     # what a perfect AI would say
cv2.rectangle(person, (30, 60), (70, 120), 1.0, -1)
cv2.circle(person, (50, 40), 18, 1.0, -1)
matte = cv2.GaussianBlur(person, (7, 7), 0)                   # soft edges, like a real matte
matte[35:45, 42:58] = 0.45                                    # the AI was unsure here (a hand near the face)
matte[10:31, 80:96] = 0.9                                     # the AI thinks the poster is a person too
blue = np.zeros((120, 100, 3), np.uint8)
blue[:] = (200, 120, 40)                                      # the ID card's blue background
print("Photo", photo.shape, "matte", matte.shape)

# ---------- PART 1: read the matte ----------
print("\nPART 1: Read the matte")
sure_person = np.sum(matte >= 0.8)
unsure = np.sum((matte >= 0.2) & (matte < 0.8))
room = np.sum(matte < 0.2)
print(f"Sure person: {sure_person}, Unsure: {unsure}, Room: {room}")

# ---------- PART 2: keep only the student ----------
print("\nPART 2: Keep only the student")
thresh = (matte > 0.35).astype(np.uint8) * 255
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

# Find the biggest component (excluding background label 0)
areas = stats[1:, cv2.CC_STAT_AREA]
biggest_label = np.argmax(areas) + 1
body = (labels == biggest_label).astype(np.float32)

print(f"Patches found: {num_labels - 1}")
print(f"Pixels body keeps: {int(np.sum(body))}")

# ---------- PART 3: paste onto the ID card ----------
print("\nPART 3: ID card")
body_3c = np.stack([body, body, body], axis=-1)
card = (photo.astype(np.float32) * body_3c + blue.astype(np.float32) * (1.0 - body_3c)).astype(np.uint8)

print("Pixel at (40, 50):", card[40, 50])
print("Pixel at (20, 88):", card[20, 88])
print("Pixel at (0, 0):", card[0, 0])

# ---------- PART 4: the ghost ----------
print("\nPART 4: Raw matte against cleaned")
matte_3c = np.stack([matte, matte, matte], axis=-1)
card_raw = (photo.astype(np.float32) * matte_3c + blue.astype(np.float32) * (1.0 - matte_3c)).astype(np.uint8)

print("Raw matte pixel at (40, 50):", card_raw[40, 50], "next to card's pixel:", card[40, 50])