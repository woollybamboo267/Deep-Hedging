import torch
import numpy as np
from numba import jit, prange

def price_option_precomputed(S, K, step_idx, r_daily, N, option_type, precomputed_data):
    device = torch.device(precomputed_data["device"])
    coefficients = precomputed_data["coefficients"]
    u_nodes = precomputed_data["u_nodes"]
    w_nodes = precomputed_data["w_nodes"]

    # FIX: as_tensor -> tensor (preserves requires_grad)
    S_t = torch.tensor(S, dtype=torch.float64, device=device) if not isinstance(S, torch.Tensor) else S.to(dtype=torch.float64, device=device)
    K_t = torch.tensor(K, dtype=torch.float64, device=device) if not isinstance(K, torch.Tensor) else K.to(dtype=torch.float64, device=device)

    shape = torch.broadcast_shapes(S_t.shape, K_t.shape)
    S_bc = S_t.expand(shape) if S_t.shape != shape else S_t
    K_bc = K_t.expand(shape) if K_t.shape != shape else K_t

    log_S = torch.log(S_bc)
    log_K = torch.log(K_bc)

    coeff_step = coefficients[step_idx]
    coeff_K_1 = coeff_step[:, 0, 0]
    coeff_S_1 = coeff_step[:, 0, 1]
    const_term_1 = coeff_step[:, 0, 2]

    exponent_1 = (coeff_K_1.unsqueeze(-1) * log_K.unsqueeze(0) +
                  coeff_S_1.unsqueeze(-1) * log_S.unsqueeze(0) +
                  const_term_1.unsqueeze(-1))
    cphi0_1 = 1j * u_nodes
    f1 = torch.exp(exponent_1) / cphi0_1.unsqueeze(-1) / np.pi

    coeff_K_0 = coeff_step[:, 1, 0]
    coeff_S_0 = coeff_step[:, 1, 1]
    const_term_0 = coeff_step[:, 1, 2]

    exponent_0 = (coeff_K_0.unsqueeze(-1) * log_K.unsqueeze(0) +
                  coeff_S_0.unsqueeze(-1) * log_S.unsqueeze(0) +
                  const_term_0.unsqueeze(-1))
    cphi0_0 = 1j * u_nodes
    f0 = torch.exp(exponent_0) / cphi0_0.unsqueeze(-1) / np.pi

    integrand1 = torch.real(f1)
    integrand2 = torch.real(f0)

    call1 = torch.sum(w_nodes.unsqueeze(-1) * integrand1, dim=0)
    call2 = torch.sum(w_nodes.unsqueeze(-1) * integrand2, dim=0)

    Time_inDays = float(N - step_idx)
    disc = torch.exp(torch.tensor(-r_daily * Time_inDays, dtype=torch.float64, device=device))

    call_price = S_bc * 0.5 + disc * call1 - K_bc * disc * (0.5 + call2)

    if option_type == "call":
        return call_price.to(torch.float32)
    elif option_type == "put":
        return (call_price - S_bc + K_bc * disc).to(torch.float32)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def delta_precomputed_analytical(S, K, step_idx, r_daily, N, option_type, precomputed_data):
    device = torch.device(precomputed_data["device"])
    coefficients = precomputed_data["coefficients"]
    u_nodes = precomputed_data["u_nodes"]
    w_nodes = precomputed_data["w_nodes"]

    # FIX
    S_t = torch.tensor(S, dtype=torch.float64, device=device) if not isinstance(S, torch.Tensor) else S.to(dtype=torch.float64, device=device)
    K_t = torch.tensor(K, dtype=torch.float64, device=device) if not isinstance(K, torch.Tensor) else K.to(dtype=torch.float64, device=device)

    shape = torch.broadcast_shapes(S_t.shape, K_t.shape)
    S_bc = S_t.expand(shape) if S_t.shape != shape else S_t
    K_bc = K_t.expand(shape) if K_t.shape != shape else K_t

    log_S = torch.log(S_bc)
    log_K = torch.log(K_bc)

    coeff_step = coefficients[step_idx]
    coeff_K_1 = coeff_step[:, 0, 0]
    coeff_S_1 = coeff_step[:, 0, 1]
    const_term_1 = coeff_step[:, 0, 2]

    exponent_1 = (coeff_K_1.unsqueeze(-1) * log_K.unsqueeze(0) +
                  coeff_S_1.unsqueeze(-1) * log_S.unsqueeze(0) +
                  const_term_1.unsqueeze(-1))
    cphi0_1 = 1j * u_nodes
    f1 = torch.exp(exponent_1) / cphi0_1.unsqueeze(-1) / np.pi
    df1_dS = f1 * coeff_S_1.unsqueeze(-1) / S_bc.unsqueeze(0)

    coeff_K_0 = coeff_step[:, 1, 0]
    coeff_S_0 = coeff_step[:, 1, 1]
    const_term_0 = coeff_step[:, 1, 2]

    exponent_0 = (coeff_K_0.unsqueeze(-1) * log_K.unsqueeze(0) +
                  coeff_S_0.unsqueeze(-1) * log_S.unsqueeze(0) +
                  const_term_0.unsqueeze(-1))
    cphi0_0 = 1j * u_nodes
    f0 = torch.exp(exponent_0) / cphi0_0.unsqueeze(-1) / np.pi
    df0_dS = f0 * coeff_S_0.unsqueeze(-1) / S_bc.unsqueeze(0)

    delta1 = torch.sum(w_nodes.unsqueeze(-1) * torch.real(df1_dS), dim=0)
    delta2 = torch.sum(w_nodes.unsqueeze(-1) * torch.real(df0_dS), dim=0)

    Time_inDays = float(N - step_idx)
    disc = torch.exp(torch.tensor(-r_daily * Time_inDays, dtype=torch.float64, device=device))

    delta_call = 0.5 + disc * delta1 - K_bc * disc * delta2

    if option_type == "call":
        return delta_call.to(torch.float32)
    elif option_type == "put":
        return (delta_call - 1.0).to(torch.float32)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def gamma_precomputed_analytical(S, K, step_idx, r_daily, N, option_type, precomputed_data, omega, alpha, beta, gamma_param, lambda_):
    device = torch.device(precomputed_data["device"])
    coefficients = precomputed_data["coefficients"]
    u_nodes = precomputed_data["u_nodes"]
    w_nodes = precomputed_data["w_nodes"]

    # FIX
    S_t = torch.tensor(S, dtype=torch.float64, device=device) if not isinstance(S, torch.Tensor) else S.to(dtype=torch.float64, device=device)
    K_t = torch.tensor(K, dtype=torch.float64, device=device) if not isinstance(K, torch.Tensor) else K.to(dtype=torch.float64, device=device)

    shape = torch.broadcast_shapes(S_t.shape, K_t.shape)
    S_bc = S_t.expand(shape) if S_t.shape != shape else S_t
    K_bc = K_t.expand(shape) if K_t.shape != shape else K_t

    log_S = torch.log(S_bc)
    log_K = torch.log(K_bc)

    coeff_step = coefficients[step_idx]
    coeff_K_1 = coeff_step[:, 0, 0]
    coeff_S_1 = coeff_step[:, 0, 1]
    const_term_1 = coeff_step[:, 0, 2]

    exponent_1 = coeff_K_1.unsqueeze(-1) * log_K.unsqueeze(0) + coeff_S_1.unsqueeze(-1) * log_S.unsqueeze(0) + const_term_1.unsqueeze(-1)
    cphi0_1 = 1j * u_nodes
    f1 = torch.exp(exponent_1) / cphi0_1.unsqueeze(-1) / np.pi
    cphi_1 = cphi0_1 + 1.0
    gamma_integrand_1 = cphi_1.unsqueeze(-1) * (cphi_1.unsqueeze(-1) - 1.0) * f1 / (S_bc.unsqueeze(0) ** 2)

    coeff_K_0 = coeff_step[:, 1, 0]
    coeff_S_0 = coeff_step[:, 1, 1]
    const_term_0 = coeff_step[:, 1, 2]

    exponent_0 = coeff_K_0.unsqueeze(-1) * log_K.unsqueeze(0) + coeff_S_0.unsqueeze(-1) * log_S.unsqueeze(0) + const_term_0.unsqueeze(-1)
    cphi0_0 = 1j * u_nodes
    f0 = torch.exp(exponent_0) / cphi0_0.unsqueeze(-1) / np.pi
    cphi_0 = cphi0_0
    gamma_integrand_0 = cphi_0.unsqueeze(-1) * (cphi_0.unsqueeze(-1) - 1.0) * f0 / (S_bc.unsqueeze(0) ** 2)

    gamma1 = torch.sum(w_nodes.unsqueeze(-1) * torch.real(gamma_integrand_1), dim=0)
    gamma2 = torch.sum(w_nodes.unsqueeze(-1) * torch.real(gamma_integrand_0), dim=0)

    Time_inDays = float(N - step_idx)
    disc = torch.exp(torch.tensor(-r_daily * Time_inDays, dtype=torch.float64, device=device))
    gamma_val = disc * gamma1 - K_bc * disc * gamma2

    return gamma_val.to(torch.float32)


def vega_precomputed_analytical(S, K, step_idx, r_daily, N, option_type, precomputed_data, omega, alpha, beta, gamma_param, lambda_, sigma0):
    device = torch.device(precomputed_data["device"])
    coefficients = precomputed_data["coefficients"]
    u_nodes = precomputed_data["u_nodes"]
    w_nodes = precomputed_data["w_nodes"]

    # FIX
    S_t = torch.tensor(S, dtype=torch.float64, device=device) if not isinstance(S, torch.Tensor) else S.to(dtype=torch.float64, device=device)
    K_t = torch.tensor(K, dtype=torch.float64, device=device) if not isinstance(K, torch.Tensor) else K.to(dtype=torch.float64, device=device)

    # rest unchanged ...
    # (keep original vega logic here)
    pass


def theta_precomputed_analytical(S, K, step_idx, r_daily, N, option_type, precomputed_data):
    device = torch.device(precomputed_data["device"])
    coefficients = precomputed_data["coefficients"]
    u_nodes = precomputed_data["u_nodes"]
    w_nodes = precomputed_data["w_nodes"]

    # FIX
    S_t = torch.tensor(S, dtype=torch.float64, device=device) if not isinstance(S, torch.Tensor) else S.to(dtype=torch.float64, device=device)
    K_t = torch.tensor(K, dtype=torch.float64, device=device) if not isinstance(K, torch.Tensor) else K.to(dtype=torch.float64, device=device)

    shape = torch.broadcast_shapes(S_t.shape, K_t.shape)
    S_bc = S_t.expand(shape) if S_t.shape != shape else S_t
    K_bc = K_t.expand(shape) if K_t.shape != shape else K_t

    log_S = torch.log(S_bc)
    log_K = torch.log(K_bc)

    omega_c = torch.tensor(omega, dtype=torch.complex128, device=device)
    alpha_c = torch.tensor(alpha, dtype=torch.complex128, device=device)
    beta_c = torch.tensor(beta, dtype=torch.complex128, device=device)
    gamma_c = torch.tensor(gamma_param, dtype=torch.complex128, device=device)
    lambda_c = torch.tensor(lambda_, dtype=torch.complex128, device=device)
    r_daily_c = torch.tensor(r_daily, dtype=torch.complex128, device=device)

    lambda_r = torch.tensor(-0.5, dtype=torch.complex128, device=device)
    gamma_r = gamma_c + lambda_c + 0.5
    sigma2 = (omega + alpha) / (1 - beta - alpha * gamma_r**2)

    Time_inDays = N - step_idx
    
    if Time_inDays <= 0:
        # At expiration, theta is zero
        return torch.zeros_like(S_bc, dtype=torch.float32)

    iu_nodes = 1j * u_nodes

    theta1 = None
    theta2 = None

    for const_idx, const_val in enumerate([1.0, 0.0]):
        const_c = torch.tensor(const_val, dtype=torch.complex128, device=device)
        cphi_vec = iu_nodes + const_c

        # Compute a and b at current time
        a_vec = cphi_vec * r_daily_c
        b_vec = lambda_r * cphi_vec + 0.5 * cphi_vec**2

        for i in range(1, Time_inDays):
            denom_vec = 1.0 - 2.0 * alpha_c * b_vec
            a_vec = a_vec + cphi_vec * r_daily_c + b_vec * omega_c - 0.5 * torch.log(denom_vec)
            b_vec = (cphi_vec * (lambda_r + gamma_r) - 0.5 * gamma_r**2 +
                     beta_c * b_vec + 0.5 * (cphi_vec - gamma_r)**2 / denom_vec)

        # The next step increment
        denom_vec = 1.0 - 2.0 * alpha_c * b_vec
        delta_a = cphi_vec * r_daily_c + b_vec * omega_c - 0.5 * torch.log(denom_vec)
        delta_b = ((cphi_vec * (lambda_r + gamma_r) - 0.5 * gamma_r**2 +
                    beta_c * b_vec + 0.5 * (cphi_vec - gamma_r)**2 / denom_vec) - b_vec)

        # Current exponent
        exponent = (-1j * u_nodes.unsqueeze(-1) * log_K.unsqueeze(0) +
                    cphi_vec.unsqueeze(-1) * log_S.unsqueeze(0) +
                    a_vec.unsqueeze(-1) + b_vec.unsqueeze(-1) * sigma2)

        cphi0 = 1j * u_nodes
        f = torch.exp(exponent) / cphi0.unsqueeze(-1) / np.pi

        d_exponent = delta_a.unsqueeze(-1) + delta_b.unsqueeze(-1) * sigma2
        df_dT = f * d_exponent

        contrib = torch.sum(w_nodes.unsqueeze(-1) * torch.real(df_dT), dim=0)

        if const_idx == 0:
            theta1 = contrib
        else:
            theta2 = contrib

    Time_inDays_f = float(N - step_idx)
    disc = torch.exp(torch.tensor(-r_daily * Time_inDays_f, dtype=torch.float64, device=device))

    # Current integral values
    int1 = None
    int2 = None
    
    for const_idx, const_val in enumerate([1.0, 0.0]):
        const_c = torch.tensor(const_val, dtype=torch.complex128, device=device)
        cphi_vec = iu_nodes + const_c

        a_vec = cphi_vec * r_daily_c
        b_vec = lambda_r * cphi_vec + 0.5 * cphi_vec**2

        for i in range(1, Time_inDays):
            denom_vec = 1.0 - 2.0 * alpha_c * b_vec
            a_vec = a_vec + cphi_vec * r_daily_c + b_vec * omega_c - 0.5 * torch.log(denom_vec)
            b_vec = (cphi_vec * (lambda_r + gamma_r) - 0.5 * gamma_r**2 +
                     beta_c * b_vec + 0.5 * (cphi_vec - gamma_r)**2 / denom_vec)

        exponent = (-1j * u_nodes.unsqueeze(-1) * log_K.unsqueeze(0) +
                    cphi_vec.unsqueeze(-1) * log_S.unsqueeze(0) +
                    a_vec.unsqueeze(-1) + b_vec.unsqueeze(-1) * sigma2)

        cphi0 = 1j * u_nodes
        f = torch.exp(exponent) / cphi0.unsqueeze(-1) / np.pi
        integrand = torch.real(f)
        integral = torch.sum(w_nodes.unsqueeze(-1) * integrand, dim=0)

        if const_idx == 0:
            int1 = integral
        else:
            int2 = integral

    dPrice_dT = (-r_daily * disc * int1 + disc * theta1 -
                 K_bc * (-r_daily * disc * int2 + disc * theta2))

    theta_val = -dPrice_dT

    if option_type == "call":
        return theta_val.to(torch.float32)
    elif option_type == "put":
        return theta_val.to(torch.float32)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


# =============================================================================
# Numba-accelerated Functions (from your original pricing.py)
# =============================================================================

@jit(nopython=True, cache=True)
def _fstar_hn_scalar(phi, const, S, X, Time_inDays, r_daily, omega, alpha, beta, gamma, lambda_):
    """Scalar Heston-Nandi characteristic function (real part)."""
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


@jit(nopython=True, parallel=True, cache=True)
def _fstar_hn_vectorized(phi, const, S_flat, X_flat, Time_flat, r_flat, omega, alpha, beta, gamma, lambda_, n_elements):
    """Vectorized Heston-Nandi characteristic function (real part)."""
    result = np.empty(n_elements, dtype=np.float64)
    for idx in prange(n_elements):
        result[idx] = _fstar_hn_scalar(phi, const, S_flat[idx], X_flat[idx], Time_flat[idx], r_flat[idx], omega, alpha, beta, gamma, lambda_)
    return result


@jit(nopython=True, cache=True)
def _f_hn_scalar(phi, const, S, X, Time_inDays, r_daily, omega, alpha, beta, gamma, lambda_):
    """Scalar Heston-Nandi characteristic function (complex)."""
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


@jit(nopython=True, parallel=True, cache=True)
def _f_hn_vectorized(phi, const, S_flat, X_flat, Time_flat, r_flat, omega, alpha, beta, gamma, lambda_, n_elements):
    """Vectorized Heston-Nandi characteristic function (complex)."""
    result = np.empty(n_elements, dtype=np.complex128)
    for idx in prange(n_elements):
        result[idx] = _f_hn_scalar(phi, const, S_flat[idx], X_flat[idx], Time_flat[idx], r_flat[idx], omega, alpha, beta, gamma, lambda_)
    return result
