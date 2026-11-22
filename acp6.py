# ================================
# ROBOT VACUUM DEPTH CHECK
# ================================

# ---------- SETUP (given to you) ----------
import cv2
import numpy as np

raw = np.tile(np.linspace(1.0, 5.0, 60, dtype=np.float32)[:, None], (1, 80))   # the floor: bigger = nearer the robot
raw[:20, :] = 1.0                                              # the far wall of the hallway
cv2.rectangle(raw, (55, 30), (70, 50), 6.0, -1)                # a toy box close to the robot
photo = np.zeros((60, 80, 3), np.uint8)
photo[:, :, 0] = np.arange(80) * 3                             # a test picture: blue = 3 x the column number
WOBBLE = 12
print("Raw depth", raw.shape, "from", raw.min(), "to", raw.max())

# ---------- PART 1: numbers from 0 to 1 ----------
print("\nPART 1: Normalise")
near = (raw - raw.min()) / (raw.max() - raw.min() + 1e-6)
print("Wall [10, 40]:", round(float(near[10, 40]), 2))
print("Floor [45, 20]:", round(float(near[45, 20]), 2))
print("Box [40, 60]:", round(float(near[40, 60]), 2))

# ---------- PART 2: where is the obstacle? ----------
print("\nPART 2: Obstacle")
close = near > 0.85
_, close_cols = np.where(close)
count = int(close.sum())
avg_col = float(close_cols.mean()) if count > 0 else float("nan")
third = photo.shape[1] / 3.0
if avg_col < third:
    side = "left"
elif avg_col < 2 * third:
    side = "middle"
else:
    side = "right"
print(f"Count: {count}, Side: {side}, Column: {avg_col:.2f}")

# ---------- PART 3: one frame of the wobble ----------
print("\nPART 3: Parallax")
h, w = near.shape
xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
shift = (WOBBLE * (near - 0.5)).astype(np.float32)
moved = cv2.remap(photo, xs + shift, ys, interpolation=cv2.INTER_LINEAR)
box_slide = float(shift[40, 60])
wall_slide = float(shift[10, 40])
box_col_now = float(moved[40, 60, 0]) / 3.0
print(f"Box [40, 60] slide: {box_slide:.2f}")
print(f"Wall [10, 40] slide: {wall_slide:.2f}")
print(f"Box [40, 60] now shows column: {box_col_now:.2f}")

# ---------- PART 4: relative, not metres ----------
print("\nPART 4: Relative depth")
raw2 = raw * 3                                                 # the same hallway, another photo: the AI's numbers are 3 times bigger
near2 = (raw2 - raw2.min()) / (raw2.max() - raw2.min() + 1e-6)
print("Raw nearest:", float(raw.max()), "| Raw2 nearest:", float(raw2.max()))
print("Same map (np.allclose):", bool(np.allclose(near, near2)))