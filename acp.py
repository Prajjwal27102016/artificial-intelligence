# ================================
# DANCE CLASS JUDGE
# ================================

# ---------- SETUP (given to you) ----------
import numpy as np

# The pose AI's arm points for one dancer at three moments, as (x, y, sure) in pixels.
# 5 and 6 = shoulders, 7 and 8 = elbows, 9 and 10 = wrists.
moments = {
    "Moment 1": {5: (220, 200, 0.9), 7: (160, 197, 0.8), 9: (100, 195, 0.9),
                 6: (300, 200, 0.9), 8: (360, 198, 0.9), 10: (420, 196, 0.8)},
    "Moment 2": {5: (220, 200, 0.9), 7: (180, 160, 0.9), 9: (150, 110, 0.7),
                 6: (300, 200, 0.9), 8: (350, 170, 0.8), 10: (400, 140, 0.9)},
    "Moment 3": {5: (220, 200, 0.9), 7: (160, 200, 0.6), 9: (100, 200, 0.1),
                 6: (300, 200, 0.9), 8: (360, 200, 0.9), 10: (420, 200, 0.2)},
}
TARGET = [180, 180, 0, 0]        # the teacher's pose, a T-pose: left upper, left lower, right upper, right lower
print("Moments:", len(moments), " target:", TARGET)

# ---------- PART 1: unsure points ----------
print("\nPART 1: Unsure points")
for name, pts in moments.items():
    unsure = sorted([k for k, (_, _, sure) in pts.items() if sure < 0.3])
    print(f"{name}: {unsure}")

# ---------- PART 2: arm angles ----------
print("\nPART 2: Arm angles")
def _seg_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return float(np.degrees(np.arctan2(dy, dx)) % 360.0)

def arm_angles(points):
    left_upper = _seg_angle(points[5], points[7])
    left_lower = _seg_angle(points[7], points[9])
    right_upper = _seg_angle(points[6], points[8])
    right_lower = _seg_angle(points[8], points[10])
    return [left_upper, left_lower, right_upper, right_lower]

for name, pts in moments.items():
    rounded = [int(round(a)) for a in arm_angles(pts)]
    print(f"{name}: {rounded}")

# ---------- PART 3: scores ----------
print("\nPART 3: Scores")
def _short_diff(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)

def score(points, target):
    if any(points[k][2] < 0.3 for k in (5, 6, 7, 8, 9, 10)):
        return 0.0
    angles = arm_angles(points)
    mean_diff = float(np.mean([_short_diff(a, t) for a, t in zip(angles, target)]))
    return max(0.0, 100.0 - mean_diff)

scores = {}
for name, pts in moments.items():
    scores[name] = int(round(score(pts, TARGET)))
print("Scores:", scores)

# ---------- PART 4: the judge ----------
print("\nPART 4: The judge")
best_moment = max(scores, key=lambda k: scores[k])
best_score = scores[best_moment]
verdict = "PASS" if best_score >= 80 else "TRY AGAIN"
print(f"{best_moment}: {best_score} -> {verdict}")