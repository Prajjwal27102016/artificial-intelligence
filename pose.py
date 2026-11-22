import os
import random
import time

import cv2
import numpy as np
import onnxruntime


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

HERE = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(HERE, "movenet.onnx")
OUTPUT_DIR = os.path.join(HERE, "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

SECONDS = 10
HOLD = 1.0
CONFIDENCE_THRESHOLD = 0.3
MATCH_THRESHOLD = 80

# ============================================================
# POSE DEFINITIONS
# Order: left upper, left lower, right upper, right lower
# ============================================================

POSES = {
    "Arms up!": [-90, -90, -90, -90],
    "T-pose!": [180, 180, 0, 0],
    "Right arm up!": [90, 90, -90, -90],
    "Left arm up!": [-90, -90, 90, 90],
    "Strong arms!": [180, -90, 0, -90],
}

# MoveNet skeleton connections
BONES = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

FONT = cv2.FONT_HERSHEY_SIMPLEX

WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
GREEN = (0, 200, 0)
ORANGE = (0, 140, 255)


# ============================================================
# LOAD MOVENET MODEL
# ============================================================

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"ONNX model not found: {MODEL_PATH}"
    )

session = onnxruntime.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)


INPUT_NAME = session.get_inputs()[0].name


# ============================================================
# BODY KEYPOINT DETECTION
# ============================================================


def find_body(frame):

    height, width = frame.shape[:2]
    side = max(height, width)

    square = cv2.copyMakeBorder(
        frame,
        0, side - height,
        0, side - width,
        cv2.BORDER_CONSTANT
    )

    image = cv2.resize(square, (192, 192))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)

    image = image.astype(np.uint8)[None, ...]

    predictions = session.run(
        ["keypoints"],
        {"pixel_values": image}
    )[0][0, 0]

    return [
        (x * side, y * side, confidence)
        for y, x, confidence in predictions
    ]




# ============================================================
# ARM ANGLE CALCULATION
# ============================================================

def arm_angles(points):

    # Identify left and right sides based on shoulder positions.
    left, right = (
        ((5, 7, 9), (6, 8, 10))
        if points[5][0] < points[6][0]
        else ((6, 8, 10), (5, 7, 9))
    )

    angles = []

    for a, b in [
        *zip(left, left[1:]),
        *zip(right, right[1:])
    ]:

        angle = np.degrees(
            np.arctan2(
                points[b][1] - points[a][1],
                points[b][0] - points[a][0]
            )
        )

        angles.append(angle)

    return angles


# ============================================================
# POSE MATCHING
# ============================================================

def match(points, target):

    required_points = (5, 6, 7, 8, 9, 10)

    if min(
        points[i][2] for i in required_points
    ) < CONFIDENCE_THRESHOLD:

        return 0

    differences = [
        abs((actual - expected + 180) % 360 - 180)
        for actual, expected in zip(
            arm_angles(points),
            target
        )
    ]

    return max(0, 100 - np.mean(differences))


# ============================================================
# DRAW TARGET POSE
# ============================================================

def draw_target(view, angles):

    x = view.shape[1] - 80
    y = 150

    cv2.rectangle(
        view,
        (x - 76, y - 70),
        (x + 76, y + 80),
        (40, 40, 40),
        -1
    )

    cv2.circle(
        view,
        (x, y - 42),
        11,
        WHITE,
        2
    )

    cv2.polylines(
        view,
        [
            np.array([
                (x - 20, y - 22),
                (x + 20, y - 22)
            ]),
            np.array([
                (x, y - 22),
                (x, y + 30)
            ])
        ],
        False,
        WHITE,
        3
    )

    def step(point, angle):

        return (
            int(point[0] + 26 * np.cos(np.radians(angle))),
            int(point[1] + 26 * np.sin(np.radians(angle)))
        )

    arms = (
        ((x - 20, y - 22), angles[:2]),
        ((x + 20, y - 22), angles[2:])
    )

    for shoulder, (upper, lower) in arms:

        elbow = step(shoulder, upper)
        hand = step(elbow, lower)

        cv2.polylines(
            view,
            [np.array([shoulder, elbow, hand])],
            False,
            YELLOW,
            4
        )


# ============================================================
# GAME INITIALIZATION
# ============================================================

def new_game():

    order = random.sample(
        list(POSES),
        len(POSES)
    )

    return order, 0, 0, None, time.time() + SECONDS


# ============================================================
# MAIN GAME LOOP
# ============================================================

def main():

    video = cv2.VideoCapture(0)

    if not video.isOpened():

        raise RuntimeError(
            "Could not open webcam. Check your camera connection."
        )

    order, turn, score, held_since, deadline = new_game()

    nice_until = 0

    try:

        while True:

            ok, frame = video.read()

            if not ok:
                break

            view = cv2.flip(frame, 1)

            points = find_body(view)

            percent = 0

            # Draw detected skeleton
            for a, b in BONES:

                if (
                    points[a][2] > CONFIDENCE_THRESHOLD
                    and points[b][2] > CONFIDENCE_THRESHOLD
                ):

                    cv2.line(
                        view,
                        (int(points[a][0]), int(points[a][1])),
                        (int(points[b][0]), int(points[b][1])),
                        YELLOW,
                        4
                    )

            # ------------------------------------------------
            # ACTIVE POSE
            # ------------------------------------------------

            if turn < len(order):

                percent = match(
                    points,
                    POSES[order[turn]]
                )

                held_since = (
                    held_since or time.time()
                    if percent >= MATCH_THRESHOLD
                    else None
                )

                completed = (
                    held_since is not None
                    and time.time() - held_since >= HOLD
                )

                expired = time.time() > deadline

                if completed or expired:

                    if completed:

                        score += 1
                        nice_until = time.time() + 0.8

                        filename = time.strftime(
                            "pose_%Y%m%d_%H%M%S.jpg"
                        )

                        cv2.imwrite(
                            os.path.join(OUTPUT_DIR, filename),
                            view
                        )

                    turn += 1

                    held_since = None

                    deadline = time.time() + SECONDS

            # ------------------------------------------------
            # TOP INFORMATION BAR
            # ------------------------------------------------

            cv2.rectangle(
                view,
                (0, 0),
                (view.shape[1], 60),
                (30, 30, 30),
                -1
            )

            if turn < len(order):

                cv2.putText(
                    view,
                    order[turn],
                    (15, 42),
                    FONT,
                    1.2,
                    YELLOW,
                    3
                )

                cv2.putText(
                    view,
                    f"{max(0, int(deadline - time.time()))} s"
                    f"    score {score}/{len(order)}",
                    (view.shape[1] - 250, 40),
                    FONT,
                    0.8,
                    WHITE,
                    2
                )

                # Match percentage bar
                cv2.rectangle(
                    view,
                    (15, 70),
                    (15 + int(3 * percent), 85),
                    GREEN if percent >= MATCH_THRESHOLD else ORANGE,
                    -1
                )

                cv2.putText(
                    view,
                    "Hold it!" if held_since else f"Match {percent:.0f}%",
                    (330, 84),
                    FONT,
                    0.6,
                    WHITE,
                    2
                )

                draw_target(
                    view,
                    POSES[order[turn]]
                )

            # ------------------------------------------------
            # GAME OVER
            # ------------------------------------------------

            else:

                view = view // 3

                cv2.putText(
                    view,
                    f"You scored {score} of {len(order)}!",
                    (40, view.shape[0] // 2),
                    FONT,
                    1.4,
                    YELLOW,
                    3
                )

                cv2.putText(
                    view,
                    "r play again    q quit",
                    (40, view.shape[0] // 2 + 50),
                    FONT,
                    0.8,
                    WHITE,
                    2
                )

            # ------------------------------------------------
            # SUCCESS FEEDBACK
            # ------------------------------------------------

            if time.time() < nice_until:

                cv2.rectangle(
                    view,
                    (0, 0),
                    (view.shape[1] - 1, view.shape[0] - 1),
                    GREEN,
                    14
                )

                cv2.putText(
                    view,
                    "Nice!",
                    (view.shape[1] // 2 - 70, view.shape[0] // 2),
                    FONT,
                    2,
                    GREEN,
                    5
                )

            # ------------------------------------------------
            # DISPLAY
            # ------------------------------------------------

            cv2.imshow("Strike a Pose", view)

            key = cv2.waitKey(1) & 255

            if (
                key in (ord("q"), 27)
                or cv2.getWindowProperty(
                    "Strike a Pose",
                    cv2.WND_PROP_VISIBLE
                ) < 1
            ):

                break

            if key == ord("r") and turn >= len(order):

                order, turn, score, held_since, deadline = new_game()

                nice_until = 0

    finally:

        video.release()
        cv2.destroyAllWindows()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

