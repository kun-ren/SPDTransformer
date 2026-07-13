import unittest

import torch
from torch import nn

from src.models.SPDTransformerClassifier import SPDTransformerClassifier
from src.models.TangentTransformerMDMClassifier import (
    TangentTransformerMDMClassifier,
)


class TangentTransformerMDMTest(unittest.TestCase):
    def make_model(self, pooling: str = "weighted") -> TangentTransformerMDMClassifier:
        return TangentTransformerMDMClassifier(
            spd_dim=4,
            num_classes=3,
            token_shape=(3, 2),
            d_model=12,
            nhead=3,
            num_layers=1,
            dim_feedforward=24,
            dropout=0.0,
            pooling=pooling,
            eps=1e-6,
        )

    def test_log_map_at_identity(self) -> None:
        model = self.make_model()
        eigenvalues = torch.tensor([1.0, 2.0, 4.0, 8.0])
        x = torch.diag(eigenvalues).reshape(1, 1, 1, 4, 4)
        x_log = model.spd_log_map(x)
        expected = torch.diag(eigenvalues.log()).reshape(1, 1, 1, 4, 4)
        self.assertTrue(torch.allclose(x_log, expected, atol=1e-6, rtol=1e-6))

    def test_isometric_vectorization_round_trip(self) -> None:
        model = self.make_model()
        matrix = torch.randn(2, 5, 4, 4)
        matrix = 0.5 * (matrix + matrix.transpose(-1, -2))
        vector = model.symmetric_matrix_to_vector(matrix)
        reconstructed = model.vector_to_symmetric_matrix(vector)

        self.assertTrue(
            torch.allclose(reconstructed, matrix, atol=1e-6, rtol=1e-6)
        )
        matrix_norm = matrix.square().sum(dim=(-2, -1))
        vector_norm = vector.square().sum(dim=-1)
        self.assertTrue(
            torch.allclose(matrix_norm, vector_norm, atol=1e-5, rtol=1e-5)
        )

    def test_forward_weighted_mdm_and_backward(self) -> None:
        torch.manual_seed(7)
        model = self.make_model(pooling="weighted")
        factor = torch.randn(2, 3, 2, 4, 4)
        identity = torch.eye(4).reshape(1, 1, 1, 4, 4)
        x = factor @ factor.transpose(-1, -2) + 0.5 * identity

        logits, aux = model(x, return_aux=True)
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertEqual(tuple(aux["token_weights"].shape), (3, 2))
        self.assertEqual(tuple(aux["pooled_log"].shape), (2, 4, 4))
        self.assertEqual(
            tuple(aux["encoded_log_tokens"].shape),
            (2, 3, 2, 4, 4),
        )
        self.assertTrue(
            torch.allclose(aux["token_weights"].sum(), torch.tensor(1.0))
        )
        self.assertTrue(
            torch.allclose(
                aux["pooled_log"],
                aux["pooled_log"].transpose(-1, -2),
                atol=1e-6,
                rtol=1e-6,
            )
        )

        logits.square().mean().backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_input_token_shape_is_checked(self) -> None:
        model = self.make_model()
        x = torch.eye(4).reshape(1, 1, 1, 4, 4).expand(2, 2, 3, 4, 4)
        with self.assertRaisesRegex(ValueError, "Expected token shape"):
            model(x)

    def test_brain_region_spdtransformer_ablation_interface(self) -> None:
        model = SPDTransformerClassifier(
            num_heads=1,
            spd_in_dim=8,
            attention_dim=[8],
            num_classes=2,
            stage_transition=True,
            time_sequence_length=2,
            frequency_sequence_length=3,
            brain_region_sequence_length=4,
            ffn_hidden_spd_dim=64,
            depth=1,
            classifier_type="mdm",
            pooling="weighted",
            dropout=0.0,
            use_position_bias=False,
            encoder_type="tangent",
        )
        factor = torch.randn(2, 2, 3, 4, 8, 8)
        x = factor @ factor.transpose(-1, -2)
        x = x + 0.5 * torch.eye(8).reshape(1, 1, 1, 1, 8, 8)

        logits, aux = model(x, return_aux=False)
        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertEqual(aux, {})
        self.assertEqual(model.encoder_type, "tangent")
        self.assertEqual(model.model.token_shape, (2, 3, 4))
        self.assertEqual(model.model.tangent_dim, 36)
        self.assertEqual(model.model.output_tangent_dim, 36)
        self.assertIsInstance(model.model.input_projection, nn.Identity)
        self.assertIsInstance(model.model.output_projection, nn.Identity)
        self.assertIsNone(model.model.position_embedding)
        self.assertEqual(
            tuple(model.model.token_weight_logits.shape),
            (2, 3, 4),
        )

    def test_omitted_singleton_region_dimension_is_supported(self) -> None:
        model = TangentTransformerMDMClassifier(
            spd_dim=3,
            num_classes=2,
            token_shape=(2, 3, 1),
            d_model=6,
            nhead=1,
            num_layers=1,
            dim_feedforward=12,
            dropout=0.0,
        )
        factor = torch.randn(2, 2, 3, 3, 3)
        x = factor @ factor.transpose(-1, -2)
        x = x + 0.5 * torch.eye(3).reshape(1, 1, 1, 3, 3)
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (2, 2))


if __name__ == "__main__":
    unittest.main()
