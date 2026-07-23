import unittest
import numpy as np
import math
from ekf_slam import EKFSLAM


class TestEKFSLAM(unittest.TestCase):
    def setUp(self):
        self.ekf = EKFSLAM()

    def test_initial_state(self):
        self.assertEqual(len(self.ekf.x), 3)
        np.testing.assert_array_equal(self.ekf.x, [0.0, 0.0, 0.0])
        self.assertEqual(self.ekf.P.shape, (3, 3))

    def test_predict(self):
        # 1 m/s straight line for 1 second
        self.ekf.predict(v=1.0, omega=0.0, dt=1.0)
        # Should be at x=1.0, y=0.0, theta=0.0
        self.assertAlmostEqual(self.ekf.x[0], 1.0)
        self.assertAlmostEqual(self.ekf.x[1], 0.0)
        self.assertAlmostEqual(self.ekf.x[2], 0.0)
        
        # Check covariance increased
        self.assertTrue(self.ekf.P[0, 0] > 1e-3)

    def test_predict_turn(self):
        # 0 m/s, turning at pi/2 rad/s for 1 second
        self.ekf.predict(v=0.0, omega=math.pi / 2.0, dt=1.0)
        self.assertAlmostEqual(self.ekf.x[2], math.pi / 2.0)

    def test_add_landmark(self):
        # Observe yellow cone at 5m directly ahead (bearing=0)
        meas = [{"range": 5.0, "bearing": 0.0, "color": "yellow"}]
        self.ekf.update(meas)

        # State should now be of size 5 (vehicle pose [3] + 1 landmark coordinates [2])
        self.assertEqual(len(self.ekf.x), 5)
        self.assertEqual(self.ekf.P.shape, (5, 5))

        # Vehicle position should still be near origin (with slight correction from Kalman update if covariance allows)
        # Landmark position should be around x=5.0, y=0.0
        self.assertAlmostEqual(self.ekf.x[3], 5.0, places=3)
        self.assertAlmostEqual(self.ekf.x[4], 0.0, places=3)
        self.assertEqual(len(self.ekf.landmarks), 1)
        self.assertEqual(self.ekf.landmarks[0]["color"], "yellow")
        self.assertEqual(self.ekf.landmarks[0]["hit_count"], 1)

    def test_associate_landmark(self):
        # 1. Add landmark at (5, 0)
        meas1 = [{"range": 5.0, "bearing": 0.0, "color": "yellow"}]
        self.ekf.update(meas1)
        self.assertEqual(len(self.ekf.landmarks), 1)

        # 2. Observe cone again at a very close position, should associate and NOT add a new landmark
        meas2 = [{"range": 5.1, "bearing": 0.01, "color": "yellow"}]
        self.ekf.update(meas2)
        self.assertEqual(len(self.ekf.landmarks), 1)
        self.assertEqual(self.ekf.landmarks[0]["hit_count"], 2)
        self.assertEqual(len(self.ekf.x), 5)

        # 3. A different colour at the same geometry is treated as corrected
        # semantic evidence, not a duplicate landmark. A geometrically distinct
        # cone must create the second landmark.
        meas3 = [{"range": 5.0, "bearing": 1.0, "color": "blue"}]
        self.ekf.update(meas3)
        self.assertEqual(len(self.ekf.landmarks), 2)
        self.assertEqual(len(self.ekf.x), 7)

    def test_unknown_landmark_is_recoloured_without_duplication(self):
        unknown = [{
            "range": 6.0,
            "bearing": 0.2,
            "color": "unknown",
            "candidate_id": 42,
        }]
        self.ekf.update(unknown)
        self.assertEqual(len(self.ekf.landmarks), 1)
        self.assertEqual(self.ekf.landmarks[0]["color"], "unknown")

        coloured = [{
            "range": 6.0,
            "bearing": 0.2,
            "color": "blue",
            "candidate_id": 42,
        }]
        self.ekf.update(coloured)
        self.assertEqual(len(self.ekf.landmarks), 1)
        self.assertEqual(self.ekf.landmarks[0]["color"], "blue")

    def test_retaining_local_candidates_compacts_dense_state(self):
        for candidate_id in range(3):
            self.ekf.update([{
                "range": 5.0 + candidate_id,
                "bearing": 0.5 * candidate_id,
                "color": "unknown",
                "candidate_id": candidate_id,
            }])
        self.ekf.retain_candidate_ids({1})
        self.assertEqual(len(self.ekf.landmarks), 1)
        self.assertEqual(self.ekf.landmarks[0]["candidate_id"], 1)
        self.assertEqual(self.ekf.x.shape, (5,))
        self.assertEqual(self.ekf.P.shape, (5, 5))


if __name__ == "__main__":
    unittest.main()
