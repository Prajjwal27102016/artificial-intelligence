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
    unsure = sorted(p for p, (_, _, sure) in pts.items() if sure < 0.3)
    print(f"{name}: {unsure}")

# ---------- PART 2: arm angles ----------
print("\nPART 2: Arm angles")
SEGMENTS = [(5, 7), (7, 9), (6, 8), (8, 10)]

def segment_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return float(np.degrees(np.arctan2(dy, dx)) % 360)

def arm_angles(points):
    return [segment_angle(points[a], points[b]) for a, b in SEGMENTS]

for name, pts in moments.items():
    angles = [round(a) for a in arm_angles(pts)]
    print(f"{name}: {angles}")

# ---------- PART 3: scores ----------
print("\nPART 3: Scores")
def angular_diff_short(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)

def score(points, target):
    if any(sure < 0.3 for _, _, sure in points.values()):
        return 0.0
    angles = arm_angles(points)
    diffs = [angular_diff_short(a, t) for a, t in zip(angles, target)]
    return max(0.0, 100.0 - float(np.mean(diffs)))

scores = {name: round(score(pts, TARGET)) for name, pts in moments.items()}
print(scores)

# ---------- PART 4: the judge ----------
print("\nPART 4: The judge")
best_moment, best_score = max(scores.items(), key=lambda item: item[1])
verdict = "PASS" if best_score >= 80 else "TRY AGAIN"
print(f"{best_moment}: {best_score} - {verdict}")