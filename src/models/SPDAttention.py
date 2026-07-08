from enum import Enum
import math
from typing import Literal

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from src.models.GeooptBiMap import GeooptBiMap


class MetricType(Enum):
    LogEuclidean = 'log-euclidean'
    LearnableMetric = 'learnable-metric'
    LearnableAffineLogFunction = 'learnable-affine-log-function'


def _symmetrize(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + x.transpose(-1, -2))


def _eye_like(x: torch.Tensor) -> torch.Tensor:
    dim = x.shape[-1]
    return torch.eye(dim, device=x.device, dtype=x.dtype)


def _safe_eigh(
        x: torch.Tensor,
        eps: float = 1e-9,
        max_tries: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = _symmetrize(x)
    if not torch.isfinite(x).all():
        raise ValueError("SPD spectral operation received NaN or Inf values.")

    try:
        return torch.linalg.eigh(x)
    except RuntimeError as error:
        last_error = error
        print(
            "[safe_eigh] torch.linalg.eigh failed without jitter; "
            f"retrying with diagonal jitter starting at {eps:.1e}. "
            f"Original error: {error}",
            flush=True,
        )

    eye = _eye_like(x)
    jitter = eps
    for attempt in range(1, max_tries + 1):
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(x + jitter * eye)
            print(
                f"[safe_eigh] succeeded on jitter attempt "
                f"{attempt}/{max_tries} with jitter={jitter:.1e}.",
                flush=True,
            )
            return eigenvalues, eigenvectors
        except RuntimeError as error:
            last_error = error
            print(
                f"[safe_eigh] jitter attempt {attempt}/{max_tries} failed "
                f"with jitter={jitter:.1e}; increasing jitter. "
                f"Error: {error}",
                flush=True,
            )
            jitter *= 10.0

    print(
        "[safe_eigh] all jitter retries failed; "
        f"trying float64 fallback with jitter={jitter:.1e}.",
        flush=True,
    )
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(
            (x + jitter * eye).to(torch.float64)
        )
        print(
            "[safe_eigh] float64 fallback succeeded "
            f"with jitter={jitter:.1e}.",
            flush=True,
        )
        return eigenvalues.to(dtype=x.dtype), eigenvectors.to(dtype=x.dtype)
    except RuntimeError as error:
        raise RuntimeError(
            "torch.linalg.eigh failed after jitter and float64 fallback. "
            "The input SPD matrix is likely severely ill-conditioned."
        ) from last_error or error


class _SPDLogEig(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, eps: float) -> torch.Tensor:
        eigenvalues, eigenvectors = _safe_eigh(x, eps=eps)
        safe_eigenvalues = eigenvalues.clamp_min(eps)
        log_eigenvalues = safe_eigenvalues.log()
        y = (
                    eigenvectors
                    * log_eigenvalues.unsqueeze(-2)
            ) @ eigenvectors.transpose(-1, -2)

        ctx.save_for_backward(safe_eigenvalues, log_eigenvalues, eigenvectors)
        ctx.eps = eps
        return _symmetrize(y)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        safe_eigenvalues, log_eigenvalues, eigenvectors = ctx.saved_tensors
        eps = ctx.eps

        grad_output = _symmetrize(grad_output)
        grad_eigenbasis = (
                eigenvectors.transpose(-1, -2)
                @ grad_output
                @ eigenvectors
        )

        lambda_i = safe_eigenvalues.unsqueeze(-1)
        lambda_j = safe_eigenvalues.unsqueeze(-2)
        log_i = log_eigenvalues.unsqueeze(-1)
        log_j = log_eigenvalues.unsqueeze(-2)

        lambda_diff = lambda_i - lambda_j
        log_diff = log_i - log_j
        scale = torch.maximum(lambda_i.abs(), lambda_j.abs()).clamp_min(eps)
        rtol = math.sqrt(torch.finfo(safe_eigenvalues.dtype).eps)
        close = lambda_diff.abs() <= rtol * scale

        safe_diff = torch.where(
            close,
            torch.ones_like(lambda_diff),
            lambda_diff,
        )
        divided_difference = log_diff / safe_diff
        limit = 2.0 / (lambda_i + lambda_j).clamp_min(eps)
        loewner = torch.where(close, limit, divided_difference)

        grad_x = (
                eigenvectors
                @ (loewner * grad_eigenbasis)
                @ eigenvectors.transpose(-1, -2)
        )
        return _symmetrize(grad_x), None


def spd_log(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    # Temporarily bypass _SPDLogEig to compare against PyTorch's native
    # torch.linalg.eigh autograd behavior.
    # return _SPDLogEig.apply(x, eps)
    x = _symmetrize(x)
    eigenvalues, eigenvectors = torch.linalg.eigh(x)
    safe_eigenvalues = eigenvalues.clamp_min(eps)
    log_eigenvalues = safe_eigenvalues.log()
    y = (
            eigenvectors
            * log_eigenvalues.unsqueeze(-2)
    ) @ eigenvectors.transpose(-1, -2)
    return _symmetrize(y)



class PositionBias(nn.Module):
    """
    Learnable relative-position bias for one attention axis.

    For query position i and key position j, the returned score bias is
    indexed by the signed relative offset j - i. This keeps the bias tied to
    relative distance instead of memorizing a separate value for every pair.
    """

    def __init__(self, max_position: int) -> None:
        super().__init__()
        if max_position < 1:
            raise ValueError(
                f"max_position must be positive, got {max_position}."
            )

        self.max_position = max_position
        self.relative_bias = nn.Parameter(
            torch.zeros(2 * self.max_position - 1)
        )

    def forward(self, attention_scope: int) -> torch.Tensor:
        if not 1 <= attention_scope <= self.max_position:
            raise ValueError(
                "attention_scope must be in "
                f"[1, {self.max_position}], got {attention_scope}."
            )

        positions = torch.arange(
            attention_scope,
            device=self.relative_bias.device,
        )
        relative_offset = positions[None, :] - positions[:, None]
        relative_index = relative_offset + self.max_position - 1
        return self.relative_bias[relative_index]


class SingleHeadAttention(nn.Module):

    def __init__(
            self,
            spd_in_dim,
            attention_dim,
            metric='log-euclidean',
            stage_transition=True,
            attention_dropout: float = 0.0,
            debug_attention_dropout: bool = False,
            learnable_metric_mode: Literal["low-rank", "kronecker"] = "low-rank",
            learnable_metric_rank: int | None = None,
            eps: float = 1e-6,
            use_position: bool = False,
            max_position: int = 128,
            debug_tensor_stats: bool = False,
    ):
        super().__init__()
        self.spd_in_dim = spd_in_dim
        self.attention_dim = attention_dim
        self.metric = MetricType(metric)
        self.stage_transition = stage_transition
        self.learnable_metric_mode = learnable_metric_mode
        self.eps = eps
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.debug_attention_dropout = debug_attention_dropout
        self.debug_tensor_stats = debug_tensor_stats
        self.affine_log_eps = eps
        self.use_position = use_position

        if self.stage_transition:
            self.query = GeooptBiMap(
                in_dim=spd_in_dim,
                out_dim=spd_in_dim,
                eps=self.eps,
                init="random",
            )
            self.key = GeooptBiMap(
                in_dim=spd_in_dim,
                out_dim=spd_in_dim,
                eps=self.eps,
                init="random",
            )
            self.value = GeooptBiMap(in_dim=spd_in_dim, out_dim=self.spd_in_dim, eps=self.eps)
        else:
            self.query = GeooptBiMap(
                in_dim=spd_in_dim,
                out_dim=attention_dim,
                eps=self.eps,
                init="random",
            )
            self.key = GeooptBiMap(
                in_dim=spd_in_dim,
                out_dim=attention_dim,
                eps=self.eps,
                init="random",
            )
            self.value = GeooptBiMap(in_dim=spd_in_dim, out_dim=self.spd_in_dim, eps=self.eps)


        tangent_feature_dim = attention_dim * (attention_dim + 1) // 2

        self.tangent_feature_dim = tangent_feature_dim
        if self.metric == MetricType.LearnableMetric:
            if learnable_metric_mode == "low-rank":
                self.learnable_metric_rank = learnable_metric_rank or min(21, tangent_feature_dim)
                self.metric_low_rank = nn.Parameter(
                    torch.randn(tangent_feature_dim, self.learnable_metric_rank) * 0.02,
                )
                nn.init.orthogonal_(self.metric_low_rank)
                self.left_metric_cholesky = None
                self.right_metric_cholesky = None
            elif learnable_metric_mode == "kronecker":  # learnable anisotropic log-map attention
                self.metric_low_rank = None
                self.left_metric_cholesky = nn.Parameter(torch.eye(attention_dim))
                self.right_metric_cholesky = nn.Parameter(torch.eye(attention_dim))
        else:
            self.metric_low_rank = None
            self.left_metric_cholesky = None
            self.right_metric_cholesky = None

        self.position_bias = (
            PositionBias(max_position=max_position)
            if self.use_position
            else None
        )

    def forward(
            self,
            x: torch.Tensor,
            return_aux: bool = True,
    ) -> tuple[Tensor, dict[str, torch.Tensor]]:

        aux = {}
        q = self.query(x)

        k = self.key(x)

        v = self.value(x)

        log_v = spd_log(v)
        log_q = spd_log(q)
        log_k = spd_log(k)

        if return_aux:
            aux['P_q'] = q
            aux['P_k'] = k
            aux['P_v'] = v

        score = self.learnableRiemannianScore(log_q, log_k)

        attention = torch.softmax(score, dim=-1)

        attention = self.attention_dropout(attention)

        weighted_log_v = torch.einsum('...ij,...jmn->...imn', attention, log_v)

        return weighted_log_v, aux

    def learnableRiemannianScore(self, log_q, log_k) -> torch.Tensor:

        if self.metric == MetricType.LogEuclidean:
            # pairwise qk score
            diff = log_q.unsqueeze(-3) - log_k.unsqueeze(-4)
            # diff [B, N_q, N_k, D, D]
            distance = torch.linalg.matrix_norm(diff, ord='fro', dim=(-2, -1))
            return distance
            # [B, N_q, N_k]

        if self.metric == MetricType.LearnableMetric:

            if self.learnable_metric_mode == "low-rank":
                q_vec = self._upper_triangular_vectorize(log_q)  # [b, s, f, tangent_dim]
                k_vec = self._upper_triangular_vectorize(log_k)

                # dot product

                q_low = torch.einsum("...d,dr->...r", q_vec, self.metric_low_rank)
                k_low = torch.einsum("...d,dr->...r", k_vec, self.metric_low_rank)

                score_low = torch.einsum("...ir,...jr->...ij", q_low, k_low)

                score_eye = torch.einsum("...id,...jd->...ij", q_vec, k_vec)

                score = score_low + self.eps * score_eye  # [b, s, f, f]
                score = score / math.sqrt(self.learnable_metric_rank)

                if self.position_bias is not None:
                    score = score + self.position_bias(score.shape[-1])

                return score


                # distance
                # q_low = torch.einsum("...d,dr->...r", q_vec, self.L)
                # k_low = torch.einsum("...d,dr->...r", k_vec, self.L)
                #
                # diff_low = q_low.unsqueeze(-2) - k_low.unsqueeze(-3)
                # dist2_low = (diff_low ** 2).sum(dim=-1)
                #
                #
                # diff = q_vec.unsqueeze(-2) - k_vec.unsqueeze(-3)
                # dist2_eye = (diff ** 2).sum(dim=-1)
                # dist2 = dist2_low + self.eps * dist2_eye
                # score = - dist2
                # score = score / math.sqrt(self.learnable_metric_rank)
                # score = - dis
                # if self.position_bias is not None:
                    # score = score + self.position_bias(score.shape[-1])
                # return score

            elif self.learnable_metric_mode == "kronecker":

                if log_q.ndim == 2 and log_k.ndim == 2:
                    diff = log_q - log_k
                else:
                    diff = log_q.unsqueeze(-3) - log_k.unsqueeze(-4)

            left_cholesky = torch.tril(self.left_metric_cholesky)
            right_cholesky = torch.tril(self.right_metric_cholesky)
            eye = torch.eye(
                self.spd_out_dim,
                device=diff.device,
                dtype=diff.dtype,
            )
            g1 = left_cholesky @ left_cholesky.transpose(-1, -2) + self.eps * eye
            g2 = right_cholesky @ right_cholesky.transpose(-1, -2) + self.eps * eye

            left_metric_diff = torch.einsum("ab,...bc->...ac", g1, diff)
            metric_diff = torch.einsum("...ab,bc->...ac", left_metric_diff, g2)
            squared_distance = (diff * metric_diff).sum(dim=(-2, -1))
            return torch.sqrt(squared_distance.clamp_min(self.eps))

        raise NotImplementedError(f"Metric {self.metric.value!r} is not implemented yet.")

    @staticmethod
    def _upper_triangular_vectorize(x: torch.Tensor) -> torch.Tensor:
        spd_dim = x.shape[-1]
        row, col = torch.triu_indices(spd_dim, spd_dim, device=x.device)
        return x[..., row, col]

    @staticmethod
    def _pairwise_squared_euclidean(
            q: torch.Tensor,
            k: torch.Tensor,
    ) -> torch.Tensor:
        q_norm = q.square().sum(dim=-1, keepdim=True)
        k_norm = k.square().sum(dim=-1).unsqueeze(-2)
        cross = torch.matmul(q, k.transpose(-1, -2))
        return (q_norm + k_norm - 2.0 * cross).clamp_min(0.0)

    def _learnable_mlp_log_function(self, x: torch.Tensor) -> torch.Tensor:
        log_x = spd_log(x)
        tangent_vector = self._upper_triangular_vectorize(log_x)
        return self.mlp_log_function(tangent_vector)

    def _spd_affine_log(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply g_theta(lambda) = a log(lambda + eps) spectrally.

        The scale is parameterized with softplus to ensure a > 0, preserving
        the ordering of eigenvalues.
        """
        eigenvalues, eigenvectors = _safe_eigh(x, eps=self.affine_log_eps)
        log_eigenvalues = (
                eigenvalues.clamp_min(0.0) + self.affine_log_eps
        ).log()

        scale = F.softplus(self.affine_log_scale_raw)
        transformed_eigenvalues = scale * log_eigenvalues

        y = (
                    eigenvectors
                    * transformed_eigenvalues.unsqueeze(-2)
            ) @ eigenvectors.transpose(-1, -2)
        return 0.5 * (y + y.transpose(-1, -2))
