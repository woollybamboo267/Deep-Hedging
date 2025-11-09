import torch
import numpy as np
import numpy as np
from numba import jit, prange
@jit(nopython=True, cache=True)
def _fstar_hn_scalar(phi, const, S, X, Time_inDays, r_daily, omega, alpha, beta, gamma, lambda_):
    cphi0 = 1j * phi
    cphi = cphi0 + const
    lambda_r = -0.5
    gamma_r = gamma + lambda_ + 0.5
    sigma2 = (omega + alpha) / (1 - beta - alpha * gamma_r**2)

    a = cphi * r_daily
    b = lambda_r * cphi + 0.5 * cphi**2

    for i in range(1, int(Time_inDays)):
        denom = 1 - 2 * alpha * b
        a = a + cphi * r_daily + b * omega - 0.5 * np.log(denom)
        b = cphi * (lambda_r + gamma_r) - 0.5 * gamma_r**2 + beta * b + 0.5 * (cphi - gamma_r)**2 / denom

    result = np.exp(-cphi0 * np.log(X) + cphi * np.log(S) + a + b * sigma2) / cphi0 / np.pi
    return result.real

# -------------------------------
# Vectorized version
@jit(nopython=True, parallel=True, cache=True)
def _fstar_hn_vectorized(phi, const, S_flat, X_flat, Time_flat, r_flat, omega, alpha, beta, gamma, lambda_, n_elements):
    result = np.empty(n_elements, dtype=np.float64)
    for idx in prange(n_elements):
        result[idx] = _fstar_hn_scalar(phi, const, S_flat[idx], X_flat[idx], Time_flat[idx], r_flat[idx], omega, alpha, beta, gamma, lambda_)
    return result

# -------------------------------
# Complex scalar
@jit(nopython=True, cache=True)
def _f_hn_scalar(phi, const, S, X, Time_inDays, r_daily, omega, alpha, beta, gamma, lambda_):
    cphi0 = 1j * phi
    cphi = cphi0 + const
    lambda_r = -0.5
    gamma_r = gamma + lambda_ + 0.5
    sigma2 = (omega + alpha) / (1 - beta - alpha * gamma_r**2)

    a = cphi * r_daily
    b = lambda_r * cphi + 0.5 * cphi**2

    for i in range(1, int(Time_inDays)):
        denom = 1 - 2 * alpha * b
        a = a + cphi * r_daily + b * omega - 0.5 * np.log(denom)
        b = cphi * (lambda_r + gamma_r) - 0.5 * gamma_r**2 + beta * b + 0.5 * (cphi - gamma_r)**2 / denom

    return np.exp(-cphi0 * np.log(X) + cphi * np.log(S) + a + b * sigma2) / cphi0 / np.pi

# -------------------------------
# Vectorized complex
@jit(nopython=True, parallel=True, cache=True)
def _f_hn_vectorized(phi, const, S_flat, X_flat, Time_flat, r_flat, omega, alpha, beta, gamma, lambda_, n_elements):
    result = np.empty(n_elements, dtype=np.complex128)
    for idx in prange(n_elements):
        result[idx] = _f_hn_scalar(phi, const, S_flat[idx], X_flat[idx], Time_flat[idx], r_flat[idx], omega, alpha, beta, gamma, lambda_)
    return result
