"""NHL rink geometry: named keypoints in real-world feet, and their positions on the 2D template.

Rink coordinates are in feet with the origin at the centre dot:
    x runs along the length of the rink, -100 (left end boards) to +100 (right end boards)
    y runs across the rink, +42.5 (far boards, top of the template) to -42.5 (near boards)
"Far" is the side of the rink away from the broadcast camera, "near" is the side closest to it.

The template image (assets/rink_template.png) is not drawn to scale: its zones are stretched
differently along each axis. rink_to_template() handles that with piecewise-linear interpolation
between markings measured on the image, so a position in feet always lands in the right place
relative to the drawn lines.
"""

from pathlib import Path

import cv2
import numpy as np

TEMPLATE_PATH = Path(__file__).parent / "assets" / "rink_template.png"

RINK_LENGTH = 200.0
RINK_WIDTH = 85.0
HALF_LENGTH = RINK_LENGTH / 2
HALF_WIDTH = RINK_WIDTH / 2

# NHL markings, in feet from centre ice.
GOAL_LINE_X = 89.0
BLUE_LINE_X = 25.0
NEUTRAL_DOT_X = 20.0
END_DOT_X = 69.0
FACEOFF_DOT_Y = 22.0
CIRCLE_RADIUS = 15.0
CORNER_RADIUS = 28.0
GOAL_POST_Y = 3.0
CREASE_RADIUS = 6.0
REF_CREASE_RADIUS = 10.0
HASH_OFFSET_X = 2.79  # hash marks are 5'7" apart, centred on the dot
HASH_LENGTH = 2.0

# Where the goal line meets the curved corner boards.
_GOAL_LINE_BOARDS_Y = (HALF_WIDTH - CORNER_RADIUS) + np.sqrt(
    CORNER_RADIUS**2 - (GOAL_LINE_X - (HALF_LENGTH - CORNER_RADIUS)) ** 2)
# Where the hash marks touch the faceoff circle, measured from the dot.
_HASH_BASE_Y = np.sqrt(CIRCLE_RADIUS**2 - HASH_OFFSET_X**2)

# Measured template pixels for each marking, used to map feet -> template pixels.
# (feet, pixel) pairs; must be sorted by feet.
_X_KNOTS = [
    (-100, 70), (-89, 145.5), (-84, 176.5), (-69, 259.5), (-54, 342.5), (-25, 475.5),
    (-20, 507.5), (-15, 558.5), (0, 642.5), (15, 726.0), (20, 779.0), (25, 811.5),
    (54, 945.5), (69, 1029.0), (84, 1112.5), (89, 1138.5), (100, 1218),
]
_Y_KNOTS = [
    (-42.5, 771), (-37, 697.5), (-22, 614), (-15, 575), (-7, 530.5), (0, 491),
    (7, 452.5), (15, 407), (22, 369), (37, 285.5), (42.5, 209),
]


def rink_to_template(points_ft):
    """Map (N, 2) rink coordinates in feet to (N, 2) template pixel coordinates."""
    pts = np.asarray(points_ft, dtype=np.float64).reshape(-1, 2)
    fx, px = zip(*_X_KNOTS)
    fy, py = zip(*_Y_KNOTS)
    return np.column_stack([np.interp(pts[:, 0], fx, px), np.interp(pts[:, 1], fy, py)])


def template_to_rink(points_px):
    """Inverse of rink_to_template()."""
    pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    fx, px = zip(*_X_KNOTS)
    fy, py = zip(*_Y_KNOTS)
    # np.interp needs increasing x; template y decreases as rink y increases.
    return np.column_stack([np.interp(pts[:, 0], px, fx), np.interp(pts[:, 1], py[::-1], fy[::-1])])


class Keypoint:
    def __init__(self, name, x, y, group, template_px=None):
        self.name = name
        self.xy = (float(x), float(y))
        self.group = group
        # Some drawn features are not to scale with the rest of the template (corner arcs, hash
        # marks), so they carry a measured pixel position instead of the interpolated one.
        self.template_px = template_px if template_px is not None else tuple(rink_to_template(self.xy)[0])

    def __repr__(self):
        return f"Keypoint({self.name!r}, x={self.xy[0]}, y={self.xy[1]})"


def _build_keypoints():
    kps = []

    def add(name, x, y, group, px=None):
        kps.append(Keypoint(name, x, y, group, px))

    # Centre ice.
    add("center_dot", 0, 0, "center")
    add("center_circle_far", 0, CIRCLE_RADIUS, "center")
    add("center_circle_near", 0, -CIRCLE_RADIUS, "center")
    add("center_circle_left", -CIRCLE_RADIUS, 0, "center")
    add("center_circle_right", CIRCLE_RADIUS, 0, "center")
    add("center_line_far_boards", 0, HALF_WIDTH, "lines")
    add("center_line_near_boards", 0, -HALF_WIDTH, "lines")

    # Referee creases (semicircles on the boards at centre ice).
    for side, sy, apex_px in (("far", 1, 265), ("near", -1, 716)):
        add(f"ref_crease_{side}_apex", 0, sy * (HALF_WIDTH - REF_CREASE_RADIUS), "center", (642.5, apex_px))
        add(f"ref_crease_{side}_left", -REF_CREASE_RADIUS, sy * HALF_WIDTH, "center", (586, 209 if sy > 0 else 771))
        add(f"ref_crease_{side}_right", REF_CREASE_RADIUS, sy * HALF_WIDTH, "center", (698, 209 if sy > 0 else 771))

    for end, sx in (("left", -1), ("right", 1)):
        e = end[0].upper()  # "L" / "R"

        # Blue line and neutral zone.
        add(f"{e}_blue_line_far_boards", sx * BLUE_LINE_X, HALF_WIDTH, "lines")
        add(f"{e}_blue_line_near_boards", sx * BLUE_LINE_X, -HALF_WIDTH, "lines")
        add(f"{e}_neutral_dot_far", sx * NEUTRAL_DOT_X, FACEOFF_DOT_Y, "neutral_dots")
        add(f"{e}_neutral_dot_near", sx * NEUTRAL_DOT_X, -FACEOFF_DOT_Y, "neutral_dots")

        # Goal line, goal and crease.
        gl_px = 145.5 if sx < 0 else 1138.5
        add(f"{e}_goal_line_far_boards", sx * GOAL_LINE_X, _GOAL_LINE_BOARDS_Y, "lines", (gl_px, 231))
        add(f"{e}_goal_line_near_boards", sx * GOAL_LINE_X, -_GOAL_LINE_BOARDS_Y, "lines", (gl_px, 748))
        add(f"{e}_goal_center", sx * GOAL_LINE_X, 0, "goal")
        add(f"{e}_goal_post_far", sx * GOAL_LINE_X, GOAL_POST_Y, "goal")
        add(f"{e}_goal_post_near", sx * GOAL_LINE_X, -GOAL_POST_Y, "goal")
        add(f"{e}_crease_apex", sx * (GOAL_LINE_X - CREASE_RADIUS), 0, "goal",
            (181 if sx < 0 else 1105, 491))

        # End boards and corners (where the straight boards meet the corner arcs).
        board_px = 70 if sx < 0 else 1218
        corner_x_px = 210 if sx < 0 else 1078
        add(f"{e}_end_boards_center", sx * HALF_LENGTH, 0, "boards")
        for side, sy in (("far", 1), ("near", -1)):
            add(f"{e}_corner_{side}_side_boards", sx * (HALF_LENGTH - CORNER_RADIUS), sy * HALF_WIDTH,
                "boards", (corner_x_px, 209 if sy > 0 else 771))
            add(f"{e}_corner_{side}_end_boards", sx * HALF_LENGTH, sy * (HALF_WIDTH - CORNER_RADIUS),
                "boards", (board_px, 350 if sy > 0 else 632))

        # End-zone faceoff circles: dot, four extreme points on the circle, four hash mark tips.
        for side, sy in (("far", 1), ("near", -1)):
            cx, cy = sx * END_DOT_X, sy * FACEOFF_DOT_Y
            p = f"{e}_{side}"
            add(f"{p}_endzone_dot", cx, cy, "endzone_dots")
            add(f"{p}_circle_top", cx, cy + CIRCLE_RADIUS, "circles")
            add(f"{p}_circle_bottom", cx, cy - CIRCLE_RADIUS, "circles")
            add(f"{p}_circle_left", cx - CIRCLE_RADIUS, cy, "circles")
            add(f"{p}_circle_right", cx + CIRCLE_RADIUS, cy, "circles")

            dot_px = rink_to_template((cx, cy))[0]
            for vert, sv in (("top", 1), ("bottom", -1)):
                tip_y = cy + sv * (_HASH_BASE_Y + HASH_LENGTH)
                tip_px_y = dot_px[1] - sv * 96  # hash tips are drawn ~96 px from the dot
                for horiz, sh in (("left", -1), ("right", 1)):
                    add(f"{p}_hash_{vert}_{horiz}", cx + sh * HASH_OFFSET_X, tip_y, "hash_marks",
                        (dot_px[0] + sh * 16.5, tip_px_y))
    return {kp.name: kp for kp in kps}


KEYPOINTS = _build_keypoints()

GROUP_COLORS = {  # BGR
    "center": (200, 120, 0),
    "lines": (0, 140, 255),
    "neutral_dots": (0, 170, 0),
    "endzone_dots": (0, 0, 200),
    "circles": (160, 0, 160),
    "hash_marks": (120, 120, 120),
    "goal": (0, 200, 200),
    "boards": (40, 40, 40),
}


def load_template():
    img = cv2.imread(str(TEMPLATE_PATH))
    if img is None:
        raise FileNotFoundError(TEMPLATE_PATH)
    return img


def draw_keypoints(img=None, names=None, labels=True, scale=1):
    """Draw keypoints (all, or just `names`) on the template. Returns a new image.

    `scale` upsizes the template first so the labels have room to be legible.
    """
    img = load_template() if img is None else img.copy()
    if scale != 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    for kp in KEYPOINTS.values():
        if names is not None and kp.name not in names:
            continue
        color = GROUP_COLORS[kp.group]
        x, y = map(int, np.round(np.array(kp.template_px) * scale))
        cv2.circle(img, (x, y), 3 * scale + 2, (255, 255, 255), -1)
        cv2.circle(img, (x, y), 3 * scale, color, -1)
        if labels:
            font_scale = 0.28 * scale
            tx = x + 4 * scale
            if kp.group == "hash_marks" and kp.name.endswith("_left"):
                # Hash marks come in close pairs: put the left one's label on its left.
                (tw, _), _ = cv2.getTextSize(kp.name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                tx = x - 4 * scale - tw
            cv2.putText(img, kp.name, (tx, y - 3 * scale), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, color, 1, cv2.LINE_AA)
    return img


if __name__ == "__main__":
    out = Path("outputs") / "rink_keypoints.png"
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), draw_keypoints(scale=3))
    print(f"{len(KEYPOINTS)} keypoints -> {out}")
