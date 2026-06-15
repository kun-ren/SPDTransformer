import unittest
from unittest.mock import patch

import torch

from src.models.SPDAttention import _SPDLogCache, spd_exp, spd_log, spd_log_cache
from src.models.SPDTransformer import SPDTransformerClassifier


def make_spd(batch_size: int = 2, dim: int = 4) -> torch.Tensor:
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(batch_size, dim, dim, generator=generator)
    eye = torch.eye(dim).expand(batch_size, dim, dim)
    return x @ x.transpose(-1, -2) + 0.5 * eye


class SPDLogCacheTest(unittest.TestCase):
    def test_reuses_log_for_the_same_tensor(self) -> None:
        x = make_spd()
        original_eigh = torch.linalg.eigh

        with patch("torch.linalg.eigh", wraps=original_eigh) as eigh:
            with spd_log_cache():
                first = spd_log(x)
                second = spd_log(x)

        self.assertIs(first, second)
        self.assertEqual(eigh.call_count, 1)

    def test_reuses_known_log_of_exponential_output(self) -> None:
        x = make_spd()
        original_eigh = torch.linalg.eigh

        with patch("torch.linalg.eigh", wraps=original_eigh) as eigh:
            with spd_log_cache():
                log_x = spd_log(x)
                reconstructed = spd_exp(log_x)
                recovered_log = spd_log(reconstructed)

        self.assertTrue(torch.allclose(recovered_log, log_x, atol=1e-6, rtol=1e-6))
        self.assertEqual(eigh.call_count, 2)

    def test_cache_is_released_between_forward_scopes(self) -> None:
        x = make_spd()
        original_eigh = torch.linalg.eigh

        with patch("torch.linalg.eigh", wraps=original_eigh) as eigh:
            with spd_log_cache():
                spd_log(x)
            with spd_log_cache():
                spd_log(x)

        self.assertEqual(eigh.call_count, 2)

    def test_cached_path_preserves_gradients(self) -> None:
        x = make_spd().requires_grad_(True)

        with spd_log_cache():
            log_x = spd_log(x)
            reconstructed = spd_exp(log_x)
            loss = spd_log(reconstructed).square().mean()

        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_classifier_forward_reduces_eigendecompositions(self) -> None:
        torch.manual_seed(11)
        model = SPDTransformerClassifier(
            spd_in_dim=4,
            spd_out_dim=4,
            num_classes=4,
            ffn_hidden_spd_dim=3,
            depth=1,
            classifier_type="pooling",
            pooling="attention",
            dropout=0.0,
        ).eval()
        x = make_spd(batch_size=2, dim=4).reshape(2, 1, 1, 4, 4)
        original_eigh = torch.linalg.eigh

        with patch("torch.linalg.eigh", wraps=original_eigh) as cached_eigh:
            cached_output = model(x)
            cached_calls = cached_eigh.call_count

        with (
            patch.object(_SPDLogCache, "get", return_value=None),
            patch.object(_SPDLogCache, "store", return_value=None),
            patch("torch.linalg.eigh", wraps=original_eigh) as uncached_eigh,
        ):
            uncached_output = model(x)
            uncached_calls = uncached_eigh.call_count

        self.assertLess(cached_calls, uncached_calls)
        self.assertTrue(torch.isfinite(cached_output).all())
        self.assertTrue(
            torch.allclose(cached_output, uncached_output, atol=2e-2, rtol=2e-2)
        )


if __name__ == "__main__":
    unittest.main()
