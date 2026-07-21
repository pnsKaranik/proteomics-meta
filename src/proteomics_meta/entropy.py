"""Shannon-style entropy features from the VAE latent space and expression profiles."""
from __future__ import annotations

import numpy as np

from .capabilities import get_logger

logger = get_logger(__name__)


def compute_latent_entropy(vaes: list, X_norm: np.ndarray) -> np.ndarray:
    """
    Closed-form differential entropy of VAE posterior q(z|x) = N(μ, σ²):
        H(z_i) = 0.5 · Σ_{d} [1 + log(2π) + log σ²_id]
    Averaged across ensemble members. Normalised to [0, 1].
    """
    log2pi        = np.log(2.0 * np.pi)
    all_entropies = []

    for vae in vaes:
        try:
            full_enc     = vae.get_full_encoder()
            outputs      = full_enc.predict(X_norm, verbose=0)
            z_log_var_np = outputs[1]
            H = 0.5 * np.sum(1.0 + log2pi + z_log_var_np, axis=1)
            all_entropies.append(H)
        except Exception as exc:
            logger.warning("Latent entropy failed for one VAE (%s).", exc)

    if not all_entropies:
        logger.warning("Latent entropy: all failed; returning zeros.")
        return np.zeros(X_norm.shape[0])

    entropy    = np.mean(all_entropies, axis=0)
    emin, emax = entropy.min(), entropy.max()
    return (entropy - emin) / (emax - emin + 1e-9)


def compute_expression_entropy(X_norm: np.ndarray) -> np.ndarray:
    """
    Shannon entropy of each protein's expression profile across samples.
        p_s = softmax(x_is),  H(i) = −Σ_s p_s · log(p_s)
    Normalised to [0, 1].
    """
    X_shifted = X_norm - X_norm.min(axis=1, keepdims=True)
    X_stable  = X_shifted - X_shifted.max(axis=1, keepdims=True)
    exp_x     = np.exp(X_stable)
    p         = exp_x / (exp_x.sum(axis=1, keepdims=True) + 1e-9)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 0, np.log(p), 0.0)
    H = -np.sum(p * log_p, axis=1)

    hmin, hmax = H.min(), H.max()
    return (H - hmin) / (hmax - hmin + 1e-9)
