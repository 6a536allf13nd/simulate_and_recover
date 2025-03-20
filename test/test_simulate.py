import unittest
import numpy as np
from src.simulate import forward_equations, inverse_equations, sample_observed_stats

class TestEZDiffusion(unittest.TestCase):
    def test_forward_equations(self):
        a, v, t = 1.2, 1.5, 0.3
        R_pred, M_pred, V_pred = forward_equations(a, v, t)
        self.assertTrue(0 <= R_pred <= 1)
        self.assertTrue(M_pred > 0)
        self.assertTrue(V_pred > 0)

    def test_inverse_equations(self):
        R_obs, M_obs, V_obs = 0.7, 0.5, 0.02
        a_est, v_est, t_est = inverse_equations(R_obs, M_obs, V_obs)
        self.assertTrue(0.5 <= a_est <= 2)
        self.assertTrue(0.5 <= v_est <= 2)
        self.assertTrue(0.1 <= t_est <= 0.5)

if __name__ == '__main__':
    unittest.main()
