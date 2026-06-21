import torch
import torch.nn as nn


class RiemannianLayerNorm(nn.Module):
    """
    Position-dependent soft spectral normalization for SPD matrices.

    Expected input shape:
        (..., attention_dim, spd_dim, spd_dim)

    For each SPD matrix X_l:

        1. Map X_l to the Log-Euclidean tangent space:
               S_l = log(X_l)

        2. Separate the isotropic log-scale:
               mu_l = trace(S_l) / n
               D_l  = S_l - mu_l * I

           D_l is symmetric and traceless.

        3. Compute the largest absolute centered log-eigenvalue:
               m_l = max_i |lambda_i(D_l)|

        4. Apply soft spectral normalization:
               Q_l = D_l / sqrt(m_l^2 + tau^2 + eps)

           Every eigenvalue of Q_l lies strictly between -1 and 1.
           Unlike exact standardization, this transformation preserves
           information about the original anisotropy magnitude.

        5. Apply a position-dependent scalar affine transformation:
               T_l = gamma_l * Q_l + beta_l * I

           All eigenvalues within one SPD matrix share the same gamma_l
           and beta_l. Different sequence positions have independent
           learnable parameters.

        6. Map back to the SPD manifold:
               Y_l = exp(T_l)

    Notes:
        - gamma_l and beta_l are unconstrained.
        - A negative gamma_l reverses the ordering of the log-eigenvalues,
          but the output remains SPD.
        - beta_l controls the overall eigenvalue scale.
        - gamma_l controls the spectral spread / anisotropy.
        - When affine=False, the output is exp(Q_l).
        - When preserve_log_mean=True, the original mean log-eigenvalue
          mu_l is added back before the matrix exponential.
    """

    def __init__(
        self,
        spd_dim: int,
        sequence_length: int,
        tau: float = 1.0,
        eps: float = 1e-6,
        affine: bool = False,
        preserve_log_mean: bool = False,
        debug_tensor_stats: bool = False,
    ):
        super().__init__()

        if spd_dim < 1:
            raise ValueError("spd_dim must be positive.")

        if sequence_length < 1:
            raise ValueError("sequence_length must be positive.")

        if tau <= 0:
            raise ValueError("tau must be strictly positive.")

        if eps < 0:
            raise ValueError("eps must be non-negative.")

        self.spd_dim = spd_dim
        self.sequence_length = sequence_length
        self.tau = tau
        self.eps = eps
        self.affine = affine
        self.preserve_log_mean = preserve_log_mean
        self.debug_tensor_stats = debug_tensor_stats

        if affine:
            # One scalar spectral gain for each absolute sequence position.
            #
            # gamma_l = 1 at initialization, so the normalized spectral
            # component is initially left unchanged.
            self.weight = nn.Parameter(torch.ones(sequence_length))

            # One scalar log-scale shift for each absolute sequence position.
            #
            # beta_l = 0 at initialization, so no additional isotropic
            # scaling is applied initially.
            self.bias = nn.Parameter(torch.zeros(sequence_length))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply soft spectral normalization.

        Args:
            x:
                logged matrices with shape
                (..., sequence_length, spd_dim, spd_dim).

        Returns:
            SPD matrices with the same shape as x.
        """

        if x.ndim < 3:
            raise ValueError(
                "Expected at least 3 dimensions: "
                "(sequence_length, spd_dim, spd_dim)."
            )

        if x.shape[-2:] != (self.spd_dim, self.spd_dim):
            raise ValueError(
                f"Expected the last two dimensions to be "
                f"({self.spd_dim}, {self.spd_dim}), "
                f"but received {tuple(x.shape)}."
            )

        current_sequence_length = x.shape[-3]

        if current_sequence_length > self.sequence_length:
            raise ValueError(
                f"Input sequence length {current_sequence_length} exceeds "
                f"the configured maximum length {self.sequence_length}."
            )


        # Construct the identity matrix on the same device and with the
        # same dtype as the input.
        eye = torch.eye(
            self.spd_dim,
            device=x.device,
            dtype=x.dtype,
        )

        # ---------------------------------------------------------------
        # Step 1: Map every SPD matrix to the symmetric tangent space.
        #
        S = x
        # Explicit symmetrization protects against small floating-point
        # asymmetries produced by eigendecomposition or reconstruction.
        S = 0.5 * (S + S.transpose(-1, -2))


        # ---------------------------------------------------------------
        # Step 2: Compute the mean log-eigenvalue of each SPD matrix.
        #
        #     mu = trace(S) / n
        #
        # Shape before unsqueeze:
        #     (..., sequence_length)
        #
        # Shape after unsqueeze:
        #     (..., sequence_length, 1, 1)
        # ---------------------------------------------------------------
        mu = S.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
        mu = mu[..., None, None]

        # Remove the isotropic log-scale.
        #
        # D is symmetric and satisfies:
        #
        #     trace(D) = 0.
        D = S - mu * eye

        D = 0.5 * (D + D.transpose(-1, -2))


        # ---------------------------------------------------------------
        # Step 3: Compute the eigenvalues of the centered log matrix.
        #
        # Since D is symmetric, eigvalsh is sufficient and more stable
        # than a general eigendecomposition.
        #
        # Shape:
        #     (..., sequence_length, spd_dim)
        # ---------------------------------------------------------------
        centered_log_eigenvalues = torch.linalg.eigvalsh(D)

        # Find the largest absolute centered log-eigenvalue:
        #
        #     m = max_i |lambda_i(D)|.
        #
        # Shape before unsqueeze:
        #     (..., sequence_length)
        #
        # Shape after unsqueeze:
        #     (..., sequence_length, 1, 1)
        spectral_radius = centered_log_eigenvalues.abs().amax(dim=-1)
        spectral_radius = spectral_radius[..., None, None]



        # ---------------------------------------------------------------
        # Step 4: Soft spectral normalization.
        #
        #     Q = D / sqrt(m^2 + tau^2 + eps)
        #
        # The eigenvalues of Q satisfy:
        #
        #     |lambda_i(Q)| < 1.
        #
        # tau determines how aggressively the spectrum is compressed:
        #
        #   - larger tau: stronger compression for ordinary inputs;
        #   - smaller tau: values approach max-absolute normalization.
        #
        # This is not exact unit-variance normalization. It preserves a
        # compressed representation of the original spectral magnitude.
        # ---------------------------------------------------------------
        denominator = torch.sqrt(
            spectral_radius.square()
            + self.tau ** 2
            + self.eps
        )

        Q = D / denominator
        Q = 0.5 * (Q + Q.transpose(-1, -2))



        # ---------------------------------------------------------------
        # Step 5: Optionally restore the original isotropic log-scale.
        #
        # preserve_log_mean=False:
        #     The input determinant information is removed.
        #
        # preserve_log_mean=True:
        #     The original mean log-eigenvalue is restored.
        #
        # Restoring mu retains the original overall covariance scale, but
        # it also means that an extremely large mu can still produce large
        # output eigenvalues.
        # ---------------------------------------------------------------
        if self.preserve_log_mean:
            T = Q + mu * eye
        else:
            T = Q

        # ---------------------------------------------------------------
        # Step 6: Apply position-dependent affine transformation.
        #
        # Each sequence position l owns two scalar parameters:
        #
        #     gamma_l = weight[l]
        #     beta_l  = bias[l]
        #
        # They are shared by all eigenvalues inside that position.
        # ---------------------------------------------------------------
        if self.affine:
            # Build a broadcast shape compatible with arbitrary leading
            # batch dimensions.
            #
            # Examples:
            #
            # x shape (B, L, n, n):
            #     parameter shape becomes (1, L, 1, 1)
            #
            # x shape (B, H, L, n, n):
            #     parameter shape becomes (1, 1, L, 1, 1)
            parameter_shape = (
                (1,) * (x.ndim - 3)
                + (current_sequence_length, 1, 1)
            )

            gamma = self.weight[
                :current_sequence_length
            ].reshape(parameter_shape)

            beta = self.bias[
                :current_sequence_length
            ].reshape(parameter_shape)

            # gamma scales the centered spectral component.
            #
            # beta * I shifts every log-eigenvalue by the same amount,
            # therefore scaling every SPD eigenvalue by exp(beta).
            if self.preserve_log_mean:
                T = gamma * Q + (mu + beta) * eye
            else:
                T = gamma * Q + beta * eye

        T = 0.5 * (T + T.transpose(-1, -2))


        # ---------------------------------------------------------------
        # Step 7: Map the symmetric matrix back to the SPD manifold.
        #
        # Every eigenvalue of exp(T) is strictly positive in exact
        # arithmetic, so the output remains SPD.
        # ---------------------------------------------------------------


        return T
