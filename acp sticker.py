# ================================
# STICKER SHOP
# ================================

# ---------- SETUP (given to you) ----------
import cv2
import numpy as np

photo = np.full((256, 512, 3), (200, 210, 220), np.uint8)            # a light table
cv2.circle(photo, (256, 150), 80, (40, 40, 200), -1)                 # a red apple
cv2.ellipse(photo, (300, 58), (30, 14), -30, 0, 360, (60, 170, 60), -1)  # its green leaf
cv2.rectangle(photo, (420, 30), (500, 110), (150, 110, 70), -1)      # a blue mug behind it
clicks = [(256, 150, 1), (460, 60, 0)]                               # click: the apple. Right-click: the mug.

guesses = np.full((3, 256, 256), -4.0, np.float32)                   # SAM's 3 guesses (256 x 256), in no order
cv2.ellipse(guesses[0], (128, 150), (40, 80), 0, 0, 360, 4.0, -1)    # the apple without its leaf
cv2.ellipse(guesses[1], (128, 150), (10, 20), 0, 0, 360, 4.0, -1)    # only a shiny spot on the apple
cv2.ellipse(guesses[2], (128, 150), (40, 80), 0, 0, 360, 4.0, -1)    # the apple with its leaf
cv2.ellipse(guesses[2], (150, 58), (15, 14), -30, 0, 360, 4.0, -1)
print("Photo", photo.shape, "guesses", guesses.shape)

# ---------- PART 1: clicks to prompts ----------
print("\nPART 1: Prompts for SAM")
scale_x = 1024.0 / photo.shape[1]  # 512 -> 1024 (2.0x)
scale_y = 1024.0 / photo.shape[0]  # 256 -> 1024 (4.0x)

points = np.array(
    [[x * scale_x, y * scale_y] for x, y, _ in clicks],
    dtype=np.float32,
)
labels = np.array([lbl for _, _, lbl in clicks], dtype=np.int64)

print("points:", points.tolist())
print("labels:", labels.tolist())
print("SAM shapes:", points[None, None].shape, labels[None, None].shape)

# ---------- PART 2: sort the guesses ----------
print("\nPART 2: Part, bigger, whole")
ordered = sorted(guesses, key=lambda g: int(np.sum(g > 0)))
sizes = [int(np.sum(g > 0)) for g in ordered]
print("sizes:", sizes)

# ---------- PART 3: the whole apple, photo-sized ----------
print("\nPART 3: Mask")
biggest = ordered[-1]
resized = cv2.resize(biggest, (photo.shape[1], photo.shape[0]), interpolation=cv2.INTER_LINEAR)
mask = (resized > 0).astype(np.uint8)
print("mask shape:", mask.shape, "pixels:", int(np.sum(mask)))

# ---------- PART 4: the sticker ----------
print("\nPART 4: Sticker")
ys, xs = np.where(mask > 0)
x0, x1 = int(xs.min()), int(xs.max()) + 1
y0, y1 = int(ys.min()), int(ys.max()) + 1
box = (x0, y0, x1, y1)

alpha = (mask * 255).astype(np.uint8)
alpha = cv2.GaussianBlur(alpha, (3, 3), 0)

bgra = np.dstack([photo, alpha])
sticker = bgra[y0:y1, x0:x1]

h, w = sticker.shape[:2]
mid_alpha = int(sticker[h // 2, w // 2, 3])
tl_alpha = int(sticker[0, 0, 3])

print("box:", box, "shape:", sticker.shape)
print("alpha middle:", mid_alpha, "top-left:", tl_alpha)
cv2.imwrite("sticker.png", sticker)