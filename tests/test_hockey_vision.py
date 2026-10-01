"""Unit tests for Hockey-Vision (rink.py, homography.py, and run_models.py).

Runs with `python -m unittest` or `pytest` without requiring GPU hardware or the gitignored
CV_Models/*.pt weights. If torch / ultralytics are not installed in the test environment,
lightweight module stubs are registered so run_models.py can be imported for testing its
pure-Python geometry and jersey-voting logic.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Register lightweight stubs when torch / ultralytics are absent so run_models imports cleanly.
if importlib.util.find_spec("torch") is None:
    torch_stub = types.ModuleType("torch")
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_stub.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    sys.modules["torch"] = torch_stub

if importlib.util.find_spec("ultralytics") is None:
    ultralytics_stub = types.ModuleType("ultralytics")
    ultralytics_stub.YOLO = object
    sys.modules["ultralytics"] = ultralytics_stub

import homography
import rink
import run_models


def _synthetic_broadcast_homography():
    """Build a realistic 1920x1080 side-view homography mapping image pixels -> rink feet."""
    rink_pts = np.float32([
        [-25.0, 42.5],
        [25.0, 42.5],
        [-25.0, -42.5],
        [25.0, -42.5],
        [0.0, 0.0],
        [-20.0, 22.0],
        [20.0, -22.0],
    ])
    img_pts = np.float32([
        [620.0, 220.0],
        [1300.0, 220.0],
        [420.0, 880.0],
        [1500.0, 880.0],
        [960.0, 500.0],
        [664.0, 336.0],
        [1318.0, 691.0],
    ])
    H, _ = cv2.findHomography(img_pts, rink_pts, 0)
    return H / H[2, 2]


class TestRinkGeometry(unittest.TestCase):
    def test_keypoints_count_and_symmetry(self):
        self.assertEqual(len(rink.KEYPOINTS), 79)
        self.assertEqual(rink.KEYPOINTS["center_dot"].xy, (0.0, 0.0))
        for name, kp in rink.KEYPOINTS.items():
            if name.startswith("L_"):
                suffix = name[2:]
                if suffix.endswith("_left"):
                    suffix = suffix[:-5] + "_right"
                elif suffix.endswith("_right"):
                    suffix = suffix[:-6] + "_left"
                mirror = rink.KEYPOINTS["R_" + suffix]
                self.assertAlmostEqual(kp.xy[0], -mirror.xy[0], places=5)
                self.assertAlmostEqual(kp.xy[1], mirror.xy[1], places=5)
            if name.endswith("_far"):
                near_name = name[:-4] + "_near"
                if near_name in rink.KEYPOINTS:
                    near_kp = rink.KEYPOINTS[near_name]
                    self.assertAlmostEqual(kp.xy[0], near_kp.xy[0], places=5)
                    self.assertAlmostEqual(kp.xy[1], -near_kp.xy[1], places=5)

    def test_rink_to_template_and_inverse_roundtrip(self):
        pts_ft = np.array([
            [0.0, 0.0],
            [-69.0, 22.0],
            [69.0, -22.0],
            [-25.0, 42.5],
            [89.0, 0.0],
            [-100.0, -42.5],
        ])
        px = rink.rink_to_template(pts_ft)
        recovered = rink.template_to_rink(px)
        np.testing.assert_allclose(recovered, pts_ft, atol=1e-5)

    def test_load_template_and_draw_keypoints(self):
        tmpl = rink.load_template()
        self.assertEqual(tmpl.ndim, 3)
        drawn = rink.draw_keypoints(tmpl, names={"center_dot", "L_far_endzone_dot"}, labels=False)
        self.assertEqual(drawn.shape, tmpl.shape)


class TestHomographyPipeline(unittest.TestCase):
    def test_dedupe_keeps_more_confident_box(self):
        boxes = [
            homography.Box({"class": "dot", "confidence": 0.60, "box_xyxy": [100, 100, 110, 110]}),
            homography.Box({"class": "dot", "confidence": 0.92, "box_xyxy": [102, 101, 112, 111]}),
            homography.Box({"class": "dot", "confidence": 0.50, "box_xyxy": [300, 300, 310, 310]}),
        ]
        kept = homography._dedupe(boxes, min_dist=15)
        self.assertEqual(len(kept), 2)
        self.assertAlmostEqual(kept[0].conf, 0.92)
        self.assertAlmostEqual(kept[1].conf, 0.50)

    def test_view_type_side_vs_end(self):
        c_far = homography.Box({"class": "Circle", "confidence": 0.9, "box_xyxy": [400, 200, 600, 260]})
        c_near = homography.Box({"class": "Circle", "confidence": 0.9, "box_xyxy": [350, 600, 700, 720]})
        self.assertEqual(homography._view_type([c_far, c_near], goal=None), "side")

        c_left = homography.Box({"class": "Circle", "confidence": 0.9, "box_xyxy": [200, 400, 500, 520]})
        c_right = homography.Box({"class": "Circle", "confidence": 0.9, "box_xyxy": [1100, 410, 1400, 530]})
        goal_end = homography.Box({"class": "Goal_Posts", "confidence": 0.9, "box_xyxy": [880, 180, 960, 240]})
        self.assertEqual(homography._view_type([c_left, c_right], goal=goal_end), "end")

    def test_circle_edge_matches_skips_clipped_sides(self):
        circle_in = homography.Box({"class": "Circle", "confidence": 0.9, "box_xyxy": [500, 300, 800, 450]})
        m_in = homography._circle_edge_matches("center", circle_in, "side", (1920, 1080))
        self.assertEqual(len(m_in), 4)

        circle_clip = homography.Box({"class": "Circle", "confidence": 0.9, "box_xyxy": [1, 300, 400, 450]})
        m_clip = homography._circle_edge_matches("L_far", circle_clip, "side", (1920, 1080))
        self.assertEqual({m.keypoint for m in m_clip},
                         {"L_far_circle_right", "L_far_circle_top", "L_far_circle_bottom"})

    def test_refine_circle_edges_moves_to_true_ellipse_extrema(self):
        H = _synthetic_broadcast_homography()
        H_inv = np.linalg.inv(H)
        angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        ring_ft = np.column_stack([15.0 * np.cos(angles), 15.0 * np.sin(angles)]).astype(np.float32)
        ring_px = cv2.perspectiveTransform(ring_ft.reshape(-1, 1, 2), H_inv).reshape(-1, 2)
        x1, y1 = ring_px.min(axis=0)
        x2, y2 = ring_px.max(axis=0)
        circle_box = homography.Box({"class": "Circle", "confidence": 0.95, "box_xyxy": [x1, y1, x2, y2]})
        center_px = tuple(cv2.perspectiveTransform(np.float32([[[0.0, 0.0]]]), H_inv)[0, 0])
        matches = homography._circle_edge_matches("center", circle_box, "side", (1920, 1080), center=center_px)
        homography._refine_circle_edges(matches, H)
        for m in matches:
            self.assertAlmostEqual(np.hypot(*m.rink_xy), rink.CIRCLE_RADIUS, places=3)
            reproj = homography.image_to_rink([m.image_xy], H)[0]
            np.testing.assert_allclose(reproj, m.rink_xy, atol=0.5)

    def test_line_candidate_resolves_correct_diagonal(self):
        H = _synthetic_broadcast_homography()
        far_px = homography.rink_to_image([[0.0, 42.5]], H)[0]
        near_px = homography.rink_to_image([[0.0, -42.5]], H)[0]
        x1, x2 = min(far_px[0], near_px[0]), max(far_px[0], near_px[0])
        y1, y2 = min(far_px[1], near_px[1]), max(far_px[1], near_px[1])
        box = homography.Box({"class": "Center_Line", "confidence": 0.9, "box_xyxy": [x1, y1, x2, y2]})
        lc = homography.LineCandidate(box, "center_line_far_boards", "center_line_near_boards")
        resolved = homography._resolve_lines([lc], H, (1920, 1080))
        self.assertEqual({m.keypoint for m in resolved},
                         {"center_line_far_boards", "center_line_near_boards"})

    def test_match_keypoints_identifies_center_and_neutral_landmarks(self):
        H = _synthetic_broadcast_homography()
        c_dot = homography.rink_to_image([[0.0, 0.0]], H)[0]
        l_far = homography.rink_to_image([[-20.0, 22.0]], H)[0]
        l_near = homography.rink_to_image([[-20.0, -22.0]], H)[0]
        dets = {
            "dots": [
                {"class": "dot", "confidence": 0.95, "box_xyxy": [c_dot[0] - 5, c_dot[1] - 5, c_dot[0] + 5, c_dot[1] + 5]},
                {"class": "dot", "confidence": 0.90, "box_xyxy": [l_far[0] - 5, l_far[1] - 5, l_far[0] + 5, l_far[1] + 5]},
                {"class": "dot", "confidence": 0.90, "box_xyxy": [l_near[0] - 5, l_near[1] - 5, l_near[0] + 5, l_near[1] + 5]},
            ],
            "rink": [
                {"class": "Circle", "confidence": 0.92, "box_xyxy": [c_dot[0] - 110, c_dot[1] - 55, c_dot[0] + 110, c_dot[1] + 55]},
                {"class": "Center_Line", "confidence": 0.90, "box_xyxy": [c_dot[0] - 15, 220, c_dot[0] + 15, 880]},
            ],
        }
        matches, lines = homography.match_keypoints(dets, (1920, 1080))
        matched_names = {m.keypoint for m in matches}
        self.assertIn("center_dot", matched_names)
        self.assertIn("L_neutral_dot_far", matched_names)
        self.assertIn("L_neutral_dot_near", matched_names)
        self.assertEqual(len(lines), 1)

    def test_compute_homography_drops_outlier_and_recovers_fit(self):
        H_true = _synthetic_broadcast_homography()
        kp_names = [
            "center_dot",
            "L_neutral_dot_far",
            "L_neutral_dot_near",
            "R_neutral_dot_far",
            "R_neutral_dot_near",
            "L_blue_line_far_boards",
            "R_blue_line_near_boards",
        ]
        matches = []
        for name in kp_names:
            uv = homography.rink_to_image([rink.KEYPOINTS[name].xy], H_true)[0]
            matches.append(homography.Match(name, uv, 1.0, "dot"))
        # Corrupt one landmark by ~200 px (~30 ft on the ice) so iterative rejection must drop it.
        bad_uv = (matches[-1].image_xy[0] + 200.0, matches[-1].image_xy[1] - 150.0)
        matches[-1] = homography.Match(matches[-1].keypoint, bad_uv, 1.0, "dot")

        H_fit, inliers = homography.compute_homography(matches, frame_size=(1920, 1080))
        self.assertIsNotNone(H_fit)
        self.assertEqual(len(inliers), len(kp_names) - 1)
        self.assertNotIn("R_blue_line_near_boards", {m.keypoint for m in inliers})
        test_pt = homography.image_to_rink(homography.rink_to_image([[10.0, -15.0]], H_true), H_fit)[0]
        np.testing.assert_allclose(test_pt, [10.0, -15.0], atol=0.25)

    def test_plausible_checks_orientation_and_scale(self):
        H_valid = _synthetic_broadcast_homography()
        self.assertTrue(homography._plausible(H_valid, (1920, 1080)))
        H_mirrored = H_valid.copy()
        H_mirrored[0, :] *= -1
        self.assertFalse(homography._plausible(H_mirrored, (1920, 1080)))
        self.assertFalse(homography._plausible(np.zeros((3, 3)), (1920, 1080)))

    def test_homography_tracker_smoothing_jump_reset_and_expiry(self):
        H1 = _synthetic_broadcast_homography()
        tracker = homography.HomographyTracker(smoothing=0.5, max_age=2, max_jump_ft=8.0)
        out1 = tracker.update(H1, (1920, 1080))
        np.testing.assert_allclose(out1, H1)

        # Small shift (< 8 ft): blended 50/50.
        H2 = H1.copy()
        H2[0, 2] += 1.0
        out2 = tracker.update(H2, (1920, 1080))
        np.testing.assert_allclose(out2, 0.5 * H1 + 0.5 * H2)

        # Missing frames: reused up to max_age=2, then reset to None.
        self.assertIsNotNone(tracker.update(None, (1920, 1080)))
        self.assertIsNotNone(tracker.update(None, (1920, 1080)))
        self.assertIsNone(tracker.update(None, (1920, 1080)))

    def test_project_players_puck_spread_and_implied_height(self):
        H = _synthetic_broadcast_homography()
        rink_positions = [(-20.0, 10.0), (-5.0, -12.0), (12.0, 8.0), (22.0, -18.0)]
        player_dets = []
        for i, xy in enumerate(rink_positions):
            u, v = homography.rink_to_image([xy], H)[0]
            player_dets.append({
                "class": "team_a_player" if i % 2 == 0 else "team_b_player",
                "confidence": 0.9,
                "box_xyxy": [u - 20, v - 95, u + 20, v],
                "jersey_number": str(10 + i),
            })
        # Add an out-of-bounds bench detection that project_players should filter out.
        u_out, v_out = homography.rink_to_image([[0.0, -55.0]], H)[0]
        player_dets.append({
            "class": "team_a_player",
            "confidence": 0.8,
            "box_xyxy": [u_out - 15, v_out - 80, u_out + 15, v_out],
        })
        puck_u, puck_v = homography.rink_to_image([[5.0, -3.0]], H)[0]
        dets = {
            "player": player_dets,
            "puck": [{"class": "puck", "confidence": 0.45, "box_xyxy": [puck_u - 4, puck_v - 4, puck_u + 4, puck_v + 4]}],
        }

        projected = homography.project_players(dets, H)
        self.assertEqual(len(projected), 4)
        self.assertFalse(homography.players_clustered(projected))

        height_ft = homography.implied_player_height(dets, H)
        self.assertIsNotNone(height_ft)
        self.assertTrue(homography.PLAYER_HEIGHT_RANGE_FT[0] <= height_ft <= homography.PLAYER_HEIGHT_RANGE_FT[1])

        puck = homography.project_puck(dets, H)
        self.assertIsNotNone(puck)
        np.testing.assert_allclose(puck["rink_xy"], [5.0, -3.0], atol=0.2)


class TestRunModelsJerseyLogic(unittest.TestCase):
    def test_assemble_number_pairs_digits_and_rejects_leading_zero(self):
        digits = [
            {"class": "8", "confidence": 0.90, "box_xyxy": [100, 120, 118, 150]},
            {"class": "7", "confidence": 0.84, "box_xyxy": [122, 121, 140, 151]},
            # Duplicate lower-confidence read overlapping "8" (suppressed by DIGIT_NMS_IOU)
            {"class": "3", "confidence": 0.55, "box_xyxy": [101, 120, 117, 149]},
            # Distant sleeve digit (ignored)
            {"class": "8", "confidence": 0.75, "box_xyxy": [20, 50, 32, 65]},
        ]
        res = run_models.assemble_number(digits)
        self.assertEqual(res[0], "87")
        self.assertAlmostEqual(res[1], 0.87, places=4)

        # Leading zero is rejected (NHL rules disallow 0 / 00).
        zero_digits = [{"class": "0", "confidence": 0.92, "box_xyxy": [100, 120, 118, 150]}]
        self.assertIsNone(run_models.assemble_number(zero_digits))

    def test_jersey_votes_partial_read_and_switch_hysteresis(self):
        votes = run_models.JerseyVotes()
        track = (7, "team_a_player")
        # One full read of "12" (0.9) + one partial single-digit read of "2" (0.8 * 0.5 = 0.4) -> 1.3 >= 1.2
        votes.add(track, ("12", 0.90))
        self.assertIsNone(votes.number(track))
        votes.add(track, ("2", 0.80))
        self.assertEqual(votes.number(track), "12")

        # A competing number "17" with score 1.5 (< 1.5 * 1.3 = 1.95) does not flip the shown number.
        votes.add(track, ("17", 0.80))
        votes.add(track, ("17", 0.70))
        self.assertEqual(votes.number(track), "12")

        # Once "17" exceeds SWITCH_RATIO * score("12"), the shown number updates.
        votes.add(track, ("17", 0.60))
        self.assertEqual(votes.number(track), "17")


if __name__ == "__main__":
    unittest.main()
