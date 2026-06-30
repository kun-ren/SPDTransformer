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
    #MonotoneNeuralSpline = 'monotone-neural-spline'
    #MLPLogFunction = 'mlp-log-function'


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


def spd_log(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    eigenvalues, eigenvectors = _safe_eigh(x, eps=eps)
    log_eigenvalues = eigenvalues.clamp_min(eps).log()
    y = (eigenvectors * log_eigenvalues.unsqueeze(-2)) @ eigenvectors.transpose(-1, -2)
    return _symmetrize(y)


def stable_attention_softmax(
    score: torch.Tensor,
    dim: int = -1,
    large_value: float = 1e6,
) -> torch.Tensor:
    score = torch.nan_to_num(
        score,
        nan=-large_value,
        posinf=large_value,
        neginf=-large_value,
    )
    score = score - score.amax(dim=dim, keepdim=True)
    attention = torch.softmax(score, dim=dim)
    attention = torch.nan_to_num(attention, nan=0.0, posinf=0.0, neginf=0.0)

    row_sum = attention.sum(dim=dim, keepdim=True)
    sequence_length = attention.shape[dim]
    uniform = torch.full_like(attention, 1.0 / sequence_length)
    attention = torch.where(
        row_sum > torch.finfo(attention.dtype).eps,
        attention / row_sum.clamp_min(torch.finfo(attention.dtype).eps),
        uniform,
    )
    return attention




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
        tangent_hidden_dim: int | None = None, # only valid when metric type is learnable-tangent-metric
        tangent_out_dim: int | None = None,
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
            self.query = GeooptBiMap(in_dim=spd_in_dim, out_dim=spd_in_dim, eps=self.eps)
            self.key = GeooptBiMap(in_dim=spd_in_dim, out_dim=spd_in_dim, eps=self.eps)
            self.value = GeooptBiMap(in_dim=spd_in_dim, out_dim=self.spd_in_dim, eps=self.eps)
        else:
            self.query = GeooptBiMap(in_dim=spd_in_dim, out_dim=attention_dim, eps=self.eps)
            self.key = GeooptBiMap(in_dim=spd_in_dim, out_dim=attention_dim, eps=self.eps)
            self.value = GeooptBiMap(in_dim=spd_in_dim, out_dim=self.spd_in_dim, eps=self.eps)

        if self.metric == MetricType.LearnableAffineLogFunction:
            # softplus(inverse_softplus(1)) = 1, so the initial transform is
            # g(lambda) = log(lambda + eps).
            inverse_softplus_one = math.log(math.expm1(1.0))
            self.affine_log_scale_raw = nn.Parameter(
                torch.tensor(inverse_softplus_one)
            )
        else:
            self.register_parameter("affine_log_scale_raw", None)

        tangent_feature_dim = attention_dim * (attention_dim + 1) // 2

        self.tangent_feature_dim = tangent_feature_dim
        if self.metric == MetricType.LearnableMetric:
            if learnable_metric_mode == "low-rank":
                learnable_metric_rank = learnable_metric_rank or min(64, tangent_feature_dim)
                self.metric_low_rank = nn.Parameter(
                    torch.empty(tangent_feature_dim, learnable_metric_rank)
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

    def forward(self, x: torch.Tensor) -> Tensor:

        aux = {}
        q = self.query(x)

        k = self.key(x)

        v = self.value(x)


        log_v = spd_log(v)

        aux['P_q'] = q
        aux['P_k'] = k
        aux['P_v'] = v


        dis = self.learnableRiemannianDistance(q, k)

        score = - dis
        if self.position_bias is not None:
            score = score + self.position_bias(dis.shape[-1])

        attention = stable_attention_softmax(score, dim=-1)

        attention_before_dropout = attention
        attention = self.attention_dropout(attention)

        self._print_attention_dropout_debug(attention_before_dropout, attention)

        weighted_log_v = torch.einsum('...ij,...jmn->...imn', attention, log_v)


        return weighted_log_v, aux

    def _print_attention_dropout_debug(
        self,
        before_dropout: torch.Tensor,
        after_dropout: torch.Tensor,
    ) -> None:
        if not self.debug_attention_dropout:
            return

        with torch.no_grad():
            dropped = (before_dropout != 0) & (after_dropout == 0)
            dropped_ratio = dropped.to(torch.float32).mean().item()
            before_row_sum = before_dropout.sum(dim=-1)
            after_row_sum = after_dropout.sum(dim=-1)

            print(
                "[AttentionDropout] "
                f"training={self.training} "
                f"p={self.attention_dropout.p:.4f} "
                f"shape={tuple(before_dropout.shape)} "
                f"dropped_ratio={dropped_ratio:.4f} "
                f"row_sum_before_mean={before_row_sum.mean().item():.4f} "
                f"row_sum_after_mean={after_row_sum.mean().item():.4f} "
                f"attn_before_min={before_dropout.min().item():.4e} "
                f"attn_before_max={before_dropout.max().item():.4e} "
                f"attn_after_min={after_dropout.min().item():.4e} "
                f"attn_after_max={after_dropout.max().item():.4e}"
            )



    def learnableRiemannianDistance(self, q, k) -> torch.Tensor:

        if self.metric == MetricType.LogEuclidean:
            log_q = spd_log(q)
            log_k = spd_log(k)

            if log_q.ndim == 2 and log_k.ndim == 2:  # condition: only a metrix
                diff = log_q - log_k
            else:  # pairwise qk score
                diff = log_q.unsqueeze(-3) - log_k.unsqueeze(-4)
                # diff [B, N_q, N_k, D, D]
            distance = torch.linalg.matrix_norm(diff, ord='fro', dim=(-2, -1))
            return distance
            # [B, N_q, N_k]

        # if self.metric == MetricType.MLPLogFunction:
        #     phi_q = self._learnable_mlp_log_function(q)
        #     phi_k = self._learnable_mlp_log_function(k)
        #
        #     if phi_q.ndim == 1 and phi_k.ndim == 1:
        #         diff = phi_q - phi_k
        #     else:
        #         diff = phi_q.unsqueeze(-2) - phi_k.unsqueeze(-3)
        #
        #     return torch.linalg.vector_norm(diff, ord=2, dim=-1)
        if self.metric == MetricType.LearnableMetric:
            log_q = spd_log(q)
            log_k = spd_log(k)

            if self.learnable_metric_mode == "low-rank":
                q_vec = self._upper_triangular_vectorize(log_q)
                k_vec = self._upper_triangular_vectorize(log_k)

                if q_vec.ndim == 1 and k_vec.ndim == 1:
                    diff = q_vec - k_vec
                else:
                    diff = q_vec.unsqueeze(-2) - k_vec.unsqueeze(-3)

                projected_diff = diff @ self.metric_low_rank
                squared_distance = (
                    projected_diff.square().sum(dim=-1)
                    + self.eps * diff.square().sum(dim=-1)
                )
                return torch.sqrt(squared_distance.clamp_min(0.0))

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
            return torch.sqrt(squared_distance.clamp_min(0.0))

        if self.metric == MetricType.LearnableAffineLogFunction:
            log_q = self._spd_affine_log(q)
            log_k = self._spd_affine_log(k)

            if log_q.ndim == 2 and log_k.ndim == 2:
                diff = log_q - log_k
            else:
                diff = log_q.unsqueeze(-3) - log_k.unsqueeze(-4)

            return torch.linalg.matrix_norm(diff, ord='fro', dim=(-2, -1))

        raise NotImplementedError(f"Metric {self.metric.value!r} is not implemented yet.")

    @staticmethod
    def _upper_triangular_vectorize(x: torch.Tensor) -> torch.Tensor:
        spd_dim = x.shape[-1]
        row, col = torch.triu_indices(spd_dim, spd_dim, device=x.device)
        return x[..., row, col]

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
