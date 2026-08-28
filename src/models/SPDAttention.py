from collections.abc import Iterable
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

POSITION_BIAS_AXES = frozenset({"time", "frequency", "region"})


def normalize_position_bias_axes(
        use_position_bias: bool,
        position_bias_axes: str | Iterable[str] | None = None,
) -> frozenset[str]:
    """Normalize the configurable attention axes that receive position bias.

    Position bias is conservative by default: when enabled without an explicit
    axis list, it is used only for time. Frequency and brain-region order do not
    generally have the same one-dimensional distance semantics as time.
    """
    if not use_position_bias:
        return frozenset()

    if position_bias_axes is None:
        return frozenset({"time"})

    if isinstance(position_bias_axes, str):
        normalized = position_bias_axes.strip().lower().replace("-", "_")
        if normalized in {"", "none", "off", "false"}:
            return frozenset()
        if normalized in {"all", "true"}:
            return POSITION_BIAS_AXES
        axes = {
            axis.strip()
            for axis in normalized.replace(";", ",").split(",")
            if axis.strip()
        }
    else:
        axes = {str(axis).strip().lower() for axis in position_bias_axes}

    unknown = axes - POSITION_BIAS_AXES
    if unknown:
        valid = ", ".join(sorted(POSITION_BIAS_AXES))
        raise ValueError(
            f"Unknown position-bias axes {sorted(unknown)}. Valid axes: {valid}."
        )
    return frozenset(axes)

def _symmetrize(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + x.transpose(-1, -2))


def _eye_like(x: torch.Tensor) -> torch.Tensor:
    dim = x.shape[-1]
    return torch.eye(dim, device=x.device, dtype=x.dtype)


def _safe_eigh(
        x: torch.Tensor,
        eps: float = 1e-9,
        max_tries: int = 6,
        check_finite: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = _symmetrize(x)
    if check_finite and not torch.isfinite(x).all():
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
    return _SPDLogEig.apply(x, eps)
    # x = _symmetrize(x)
    # eigenvalues, eigenvectors = torch.linalg.eigh(x)
    # safe_eigenvalues = eigenvalues.clamp_min(eps)
    # log_eigenvalues = safe_eigenvalues.log()
    # y = (
    #         eigenvectors
    #         * log_eigenvalues.unsqueeze(-2)
    # ) @ eigenvectors.transpose(-1, -2)
    # return _symmetrize(y)


def spd_exp(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Exponentiate symmetric log-SPD matrices without float overflow.

    The infinity norm bounds every eigenvalue of a symmetric matrix. Rescaling
    only matrices outside the usable log range keeps ``matrix_exp`` finite and
    avoids silently replacing NaN or Inf values after they have appeared.
    """

    if not 0.0 < eps < 1.0:
        raise ValueError(f"spd_exp eps must be in (0, 1), got {eps}.")
    x = _symmetrize(x)
    if not torch.isfinite(x).all():
        raise RuntimeError("Non-finite log-SPD value detected before matrix_exp.")

    dimension = x.shape[-1]
    finfo = torch.finfo(x.dtype)
    numeric_log_limit = (
        math.log(finfo.max) - math.log(float(dimension)) - 2.0
    )
    usable_log_limit = max(1.0, min(-math.log(eps), numeric_log_limit))
    infinity_norm = x.abs().sum(dim=-1).amax(dim=-1, keepdim=True).unsqueeze(-1)
    compression = (infinity_norm / usable_log_limit).clamp_min(1.0)
    stabilized_x = x / compression

    output = torch.matrix_exp(stabilized_x.contiguous())
    if not torch.isfinite(output).all():
        raise RuntimeError(
            "Non-finite SPD value detected after stabilized matrix_exp."
        )
    return _symmetrize(output)



class PositionBias(nn.Module):
    """
    Learnable relative-position bias for one attention axis.

    The effective bias is exactly zero at initialization, so enabling this
    module starts from the no-position-bias model. A learnable ReZero-style
    gate can then admit a symmetric locality prior when it helps the task.
    """

    def __init__(
            self,
            max_position: int,
            max_bias: float = 0.5,
    ) -> None:
        super().__init__()
        if max_position < 1:
            raise ValueError(
                f"max_position must be positive, got {max_position}."
            )
        if max_bias <= 0:
            raise ValueError(f"max_bias must be positive, got {max_bias}.")

        self.max_position = max_position
        self.max_bias = float(max_bias)
        distance = torch.arange(max_position, dtype=torch.get_default_dtype())
        self.distance_bias = nn.Parameter(-distance / max(max_position - 1, 1))
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, attention_scope: int) -> torch.Tensor:
        if not 1 <= attention_scope <= self.max_position:
            raise ValueError(
                "attention_scope must be in "
                f"[1, {self.max_position}], got {attention_scope}."
            )

        positions = torch.arange(
            attention_scope,
            device=self.distance_bias.device,
        )
        distance = (positions[None, :] - positions[:, None]).abs()
        distance_values = (
            self.max_bias
            * torch.tanh(self.gate)
            * torch.tanh(self.distance_bias[:attention_scope])
        )
        return distance_values[distance]


class SingleHeadAttention(nn.Module):

    def __init__(
            self,
            spd_in_dim,
            attention_dim=None,
            metric='log-euclidean',
            stage_transition=True,
            attention_dropout: float = 0.0,
            debug_attention_dropout: bool = False,
            learnable_metric_mode: Literal["full", "low-rank"] = "low-rank",
            learnable_metric_score: Literal["qgk", "distance"] = "qgk",
            learnable_metric_rank: int | None = None,
            eps: float = 1e-6,
            use_position: bool = False,
            max_position: int = 128,
            position_bias_max: float = 0.5,
            attention_score_target_rms: float = 1.0,
            attention_score_clip: float = 5.0,
            debug_tensor_stats: bool = False,
            spd_out_dim: int | None = None,
            metric_eps: float | None = None,
    ):
        super().__init__()
        if attention_dim is None:
            if spd_out_dim is None:
                raise ValueError("attention_dim must be provided.")
            attention_dim = int(spd_out_dim)
        elif spd_out_dim is not None and int(spd_out_dim) != int(attention_dim):
            raise ValueError(
                "attention_dim and legacy spd_out_dim must match when both "
                f"are provided, got {attention_dim} and {spd_out_dim}."
            )
        if metric_eps is not None:
            eps = float(metric_eps)

        self.spd_in_dim = spd_in_dim
        self.attention_dim = attention_dim
        self.metric = MetricType(metric)
        self.stage_transition = stage_transition
        normalized_metric_mode = (
            str(learnable_metric_mode).strip().lower().replace("_", "-")
        )
        if normalized_metric_mode in {"matrix", "full-rank", "unparameterized"}:
            normalized_metric_mode = "full"
        if normalized_metric_mode not in {"full", "low-rank"}:
            raise ValueError(
                "learnable_metric_mode must be 'full' or 'low-rank', got "
                f"{learnable_metric_mode!r}."
            )
        normalized_metric_score = (
            str(learnable_metric_score).strip().lower().replace("_", "-")
        )
        if normalized_metric_score in {"dot", "dot-product", "bilinear", "similarity"}:
            normalized_metric_score = "qgk"
        if normalized_metric_score in {"dist", "squared-distance", "distance-squared"}:
            normalized_metric_score = "distance"
        if normalized_metric_score not in {"qgk", "distance"}:
            raise ValueError(
                "learnable_metric_score must be 'qgk' or 'distance', got "
                f"{learnable_metric_score!r}."
            )
        self.learnable_metric_mode = normalized_metric_mode
        self.learnable_metric_score = normalized_metric_score
        self.eps = eps
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.debug_attention_dropout = debug_attention_dropout
        self.debug_tensor_stats = debug_tensor_stats
        self.affine_log_eps = eps
        self.use_position = use_position
        if attention_score_target_rms <= 0:
            raise ValueError(
                "attention_score_target_rms must be positive, got "
                f"{attention_score_target_rms}."
            )
        if attention_score_clip <= 0:
            raise ValueError(
                f"attention_score_clip must be positive, got {attention_score_clip}."
            )
        self.attention_score_target_rms = float(attention_score_target_rms)
        self.attention_score_clip = float(attention_score_clip)

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

        if self.metric == MetricType.LearnableAffineLogFunction:
            inverse_softplus_one = math.log(math.expm1(1.0))
            self.affine_log_scale_raw = nn.Parameter(
                torch.tensor(inverse_softplus_one)
            )
        else:
            self.register_parameter("affine_log_scale_raw", None)

        tangent_feature_dim = attention_dim * (attention_dim + 1) // 2

        self.tangent_feature_dim = tangent_feature_dim
        self.learnable_metric_rank = None
        self.metric_matrix = None
        self.metric_low_rank = None
        if self.metric == MetricType.LearnableMetric:
            if self.learnable_metric_mode == "full":
                # Direct, unconstrained dense G in tangent-vector coordinates.
                # It is symmetrized at use time but receives no factorization.
                self.metric_matrix = nn.Parameter(torch.eye(tangent_feature_dim))
            else:
                rank = (
                    min(21, tangent_feature_dim)
                    if learnable_metric_rank is None or int(learnable_metric_rank) <= 0
                    else int(learnable_metric_rank)
                )
                if rank > tangent_feature_dim:
                    raise ValueError(
                        "learnable_metric_rank must not exceed tangent feature "
                        f"dimension {tangent_feature_dim}, got {rank}."
                    )
                self.learnable_metric_rank = rank
                self.metric_low_rank = nn.Parameter(
                    torch.randn(tangent_feature_dim, self.learnable_metric_rank) * 0.02,
                )
                nn.init.orthogonal_(self.metric_low_rank)

        self.position_bias = (
            PositionBias(max_position=max_position, max_bias=position_bias_max)
            if self.use_position
            else None
        )
        row, col = torch.triu_indices(attention_dim, attention_dim)
        self.register_buffer("triu_row", row, persistent=False)
        self.register_buffer("triu_col", col, persistent=False)
        vector_scale = torch.ones(row.numel())
        vector_scale[row != col] = math.sqrt(2.0)
        self.register_buffer("triu_scale", vector_scale, persistent=False)

    def forward(
            self,
            x: torch.Tensor,
            return_aux: bool = True,
    ) -> tuple[Tensor, dict[str, torch.Tensor]]:

        aux = {}
        q = self.query(x)

        k = self.key(x)

        v = self.value(x)

        log_v = spd_log(v, eps=self.eps)
        log_q = spd_log(q, eps=self.eps)
        log_k = spd_log(k, eps=self.eps)

        if return_aux:
            aux['P_q'] = q
            aux['P_k'] = k
            aux['P_v'] = v

        score = self.learnableRiemannianScore(log_q, log_k)
        if self.position_bias is not None:
            score = score + self.position_bias(score.shape[-1])
        score = self._stabilize_attention_score(score)

        # Metric factors are RMS-capped before score construction and the
        # score is normalized with an overflow-safe RMS below. Keep the cast
        # explicit in case a caller supplies a higher-precision score tensor.
        attention = torch.softmax(score, dim=-1).to(dtype=log_v.dtype)

        attention = self.attention_dropout(attention)

        weighted_log_v = torch.einsum('...ij,...jmn->...imn', attention, log_v)

        return weighted_log_v, aux

    def learnableRiemannianScore(self, log_q, log_k) -> torch.Tensor:

        if self.metric == MetricType.LogEuclidean:
            # pairwise qk score
            diff = log_q.unsqueeze(-3) - log_k.unsqueeze(-4)
            # diff [B, N_q, N_k, D, D]
            distance = torch.linalg.matrix_norm(diff, ord='fro', dim=(-2, -1))
            return - distance / 5
            # [B, N_q, N_k]

        if self.metric == MetricType.LearnableMetric:
            q_vec = self._upper_triangular_vectorize_cached(log_q)
            k_vec = self._upper_triangular_vectorize_cached(log_k)

            if self.learnable_metric_mode == "full":
                score = self._full_metric_score(q_vec, k_vec)
            else:
                score = self._low_rank_metric_score(q_vec, k_vec)

            return score

        raise NotImplementedError(f"Metric {self.metric.value!r} is not implemented yet.")

    def _stabilize_attention_score(self, score: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(score).all():
            finite = torch.isfinite(score)
            finite_count = int(finite.sum().item())
            finite_values = score[finite]
            finite_range = (
                "none"
                if finite_values.numel() == 0
                else (
                    f"[{finite_values.min().item():.3e}, "
                    f"{finite_values.max().item():.3e}]"
                )
            )
            raise RuntimeError(
                "Non-finite SPD attention score detected before softmax. "
                f"finite={finite_count}/{score.numel()}, "
                f"finite_range={finite_range}. Check SPD eigenvalue range, "
                "eps, and attention learning rate."
            )
        score = score - score.mean(dim=-1, keepdim=True)
        # Compute RMS after scaling by the row maximum. This is equivalent to
        # sqrt(mean(score**2)) but cannot overflow on finite float32 scores.
        row_scale = score.abs().amax(dim=-1, keepdim=True)
        safe_row_scale = row_scale.clamp_min(torch.finfo(score.dtype).tiny)
        scaled_score = score / safe_row_scale
        row_rms = (
            safe_row_scale
            * scaled_score.square().mean(dim=-1, keepdim=True).sqrt()
        )
        compression = (
            row_rms / self.attention_score_target_rms
        ).clamp_min(1.0)
        score = score / compression
        score = score.clamp(
            min=-self.attention_score_clip,
            max=self.attention_score_clip,
        )
        return score - score.amax(dim=-1, keepdim=True)

    @staticmethod
    def _upper_triangular_vectorize(x: torch.Tensor) -> torch.Tensor:
        spd_dim = x.shape[-1]
        row, col = torch.triu_indices(spd_dim, spd_dim, device=x.device)
        scale = torch.ones(row.numel(), device=x.device, dtype=x.dtype)
        scale[row != col] = math.sqrt(2.0)
        return x[..., row, col] * scale

    def _upper_triangular_vectorize_cached(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.attention_dim, self.attention_dim):
            raise ValueError(
                "Expected log-SPD matrix shape "
                f"(..., {self.attention_dim}, {self.attention_dim}), "
                f"got {tuple(x.shape)}."
            )
        return x[..., self.triu_row, self.triu_col] * self.triu_scale.to(
            dtype=x.dtype
        )

    def _full_metric_score(
            self,
            q_vec: torch.Tensor,
            k_vec: torch.Tensor,
    ) -> torch.Tensor:
        if self.metric_matrix is None:
            raise RuntimeError("Full metric score requires metric_matrix G.")

        metric_matrix = self._cap_metric_rms(self.metric_matrix)
        metric = 0.5 * (
            metric_matrix + metric_matrix.transpose(-1, -2)
        )
        scale = math.sqrt(self.tangent_feature_dim)

        if self.learnable_metric_score == "qgk":
            q_metric = torch.einsum("...id,df->...if", q_vec, metric)
            score = torch.einsum("...id,...jd->...ij", q_metric, k_vec)
            return score / scale

        diff = q_vec.unsqueeze(-2) - k_vec.unsqueeze(-3)
        metric_diff = torch.einsum("...ijd,df->...ijf", diff, metric)
        squared_distance = (diff * metric_diff).sum(dim=-1)
        return -squared_distance / scale

    def _low_rank_metric_score(
            self,
            q_vec: torch.Tensor,
            k_vec: torch.Tensor,
    ) -> torch.Tensor:
        if self.metric_low_rank is None or self.learnable_metric_rank is None:
            raise RuntimeError("Low-rank metric score requires factor L and rank r.")

        metric_low_rank = self._cap_metric_rms(self.metric_low_rank)
        scale = math.sqrt(self.learnable_metric_rank)
        if self.learnable_metric_score == "qgk":
            q_low = torch.einsum("...d,dr->...r", q_vec, metric_low_rank)
            k_low = torch.einsum("...d,dr->...r", k_vec, metric_low_rank)
            score_low = torch.einsum("...ir,...jr->...ij", q_low, k_low)
            score_eye = torch.einsum("...id,...jd->...ij", q_vec, k_vec)
            return (score_low + self.eps * score_eye) / scale

        diff = q_vec.unsqueeze(-2) - k_vec.unsqueeze(-3)
        diff_low = torch.einsum("...ijd,dr->...ijr", diff, metric_low_rank)
        squared_distance_low = diff_low.square().sum(dim=-1)
        squared_distance_eye = diff.square().sum(dim=-1)
        squared_distance = (
            squared_distance_low + self.eps * squared_distance_eye
        )
        return -squared_distance / scale

    @staticmethod
    def _cap_metric_rms(metric: torch.Tensor) -> torch.Tensor:
        """Cap redundant global metric scale without changing its direction.

        Attention rows are already RMS-compressed before softmax, so a global
        metric scale above one carries no useful contrast information. The
        small float64 reduction avoids overflow while the large score tensors
        remain in their efficient training dtype.
        """

        rms = metric.to(torch.float64).square().mean().sqrt()
        scale = rms.clamp_min(1.0).to(device=metric.device, dtype=metric.dtype)
        return metric / scale

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
