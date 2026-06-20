import math
import unittest

import torch
import torch.nn.functional as F

from src.models.SPDAttention import SingleHeadAttention


class LearnableAffineLogTest(unittest.TestCase):
    def setUp(self) -> None:
        self.eps = 1e-6
        self.attention = SingleHeadAttention(
            spd_in_dim=2,
            spd_out_dim=2,
            metric="learnable-affine-log-function",
            metric_eps=self.eps,
        )

    def test_initial_transform_is_stabilized_matrix_log(self) -> None:
        x = torch.diag(torch.tensor([1.0, 4.0]))

        transformed = self.attention._spd_affine_log(x)
        expected = torch.diag(torch.log(torch.tensor([1.0, 4.0]) + self.eps))

        scale = F.softplus(self.attention.affine_log_scale_raw)
        self.assertAlmostEqual(scale.item(), 1.0, places=6)
        self.assertTrue(torch.allclose(transformed, expected, atol=1e-6, rtol=1e-6))

    def test_transform_matches_scaled_spectral_log_formula(self) -> None:
        scale = 2.0
        inverse_softplus_scale = math.log(math.expm1(scale))
        with torch.no_grad():
            self.attention.affine_log_scale_raw.fill_(inverse_softplus_scale)

        x = torch.diag(torch.tensor([0.5, 3.0]))
        transformed = self.attention._spd_affine_log(x)
        expected_eigenvalues = scale * torch.log(
            torch.tensor([0.5, 3.0]) + self.eps
        )

        self.assertTrue(
            torch.allclose(
                transformed,
                torch.diag(expected_eigenvalues),
                atol=1e-5,
                rtol=1e-5,
            )
        )

    def test_scale_remains_positive(self) -> None:
        with torch.no_grad():
            self.attention.affine_log_scale_raw.fill_(-20.0)

        scale = F.softplus(self.attention.affine_log_scale_raw)
        self.assertGreater(scale.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
