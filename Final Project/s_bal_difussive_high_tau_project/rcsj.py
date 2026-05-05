"""
Note for publish-ready code:
1. Remove the long ballistic part
2. Remove the RK4 deterministic solver (keep only the SDE solver)


Unified RCSJ-with-quasiparticle-shunt model, with modular CPR and F(v)
selectors for different junction physics regimes.

Equation of motion (dimensionless):
    phi_ddot + F(phi_dot/Q; F_code, i_exc) + I_s(phi; cpr_code, p) = i + xi

CPR codes (with peak normalized to 1):
    0 = sinusoidal           tunnel junction
    1 = Kulik-Omelyanchuk-1  short ballistic, parameter = transparency tau
    2 = long ballistic       parameter = L/xi (junction length / coherence length)
    3 = Kulik-Omelyanchuk-2  short diffusive (no shape parameter)

F(v) codes:
    0 = linear              F(v) = v   (tunnel, long-ballistic-like dissipation)
    1 = BTK with i_exc      F(v) = v + i_exc * tanh(v / w)

The two are independent: any CPR can pair with any F. In the dataset we use
specific physical combinations:
    A: cpr=0, F=0 (tunnel)
    B: cpr=1, F=1 (KO-1 short ballistic + BTK)
    C: cpr=2, F=0 (long ballistic, no excess current)
    D: cpr=3, F=1 (KO-2 short diffusive + BTK)

CPR mathematical forms (unnormalized; peak normalization is applied via
inv_peak):

  cpr=0 (sin):     I_s(phi) = sin(phi)

  cpr=1 (KO-1):    I_s(phi) = sin(phi) / sqrt(1 - tau * sin^2(phi/2))
                    (Kulik & Omelyanchuk, single ballistic channel)

  cpr=2 (long ballistic):
                    Truncated harmonic sum approximating Ishii's result for
                    a long ballistic SNS junction at T=0. We parametrize the
                    crossover by L/xi, with a damping factor exp(-n*L/xi):

                    I_s(phi) = sum_{n=1..N} (-1)^{n+1}/n * exp(-n*L/xi) * sin(n*phi)

                    For L/xi -> 0  -> sinusoidal (n=1 dominant).
                    For L/xi -> infty -> sawtooth-like, peak shifts toward pi.

  cpr=3 (KO-2):     I_s(phi) = cos(phi/2) * arctanh(sin(phi/2))
                    (Kulik & Omelyanchuk, short diffusive, AVZ form)
                    Logarithmic-cusp peak.

References:
    Likharev, Rev. Mod. Phys. 51, 101 (1979)
    Golubov, Kupriyanov, Il'ichev, Rev. Mod. Phys. 76, 411 (2004)
"""
import numpy as np
from numba import njit, prange


N_HARMONICS_LONG_BAL = 12   # truncation of the harmonic sum for cpr=2


# ---------------------------------------------------------------------------
# CPR evaluation (Numba-jitted, scalar version)
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True, inline='always')
def _cpr_unnorm(phi, cpr_code, p):
    """Unnormalized CPR. p is the shape parameter:
       cpr_code=1: tau (transparency)
       cpr_code=2: L/xi
       cpr_code=0,3: ignored
    """
    if cpr_code == 0:
        return np.sin(phi)
    elif cpr_code == 1:
        s = np.sin(phi * 0.5)
        return np.sin(phi) / np.sqrt(1.0 - p * s * s)
    elif cpr_code == 2:
        # Harmonic sum: sum_n (-1)^(n+1)/n * exp(-n*L/xi) * sin(n*phi)
        result = 0.0
        for n in range(1, N_HARMONICS_LONG_BAL + 1):
            sign = 1.0 if (n % 2 == 1) else -1.0
            result += sign / n * np.exp(-n * p) * np.sin(n * phi)
        return result
    elif cpr_code == 3:
        # KO-2 short diffusive: cos(phi/2) * arctanh(sin(phi/2))
        s = np.sin(phi * 0.5)
        c = np.cos(phi * 0.5)
        # Clamp |s| < 1 to avoid arctanh blowup right at phi = pi
        s_safe = min(abs(s), 0.99999) * (1.0 if s >= 0 else -1.0)
        return c * np.arctanh(s_safe)
    else:
        return 0.0


@njit(cache=True, fastmath=True, inline='always')
def _cpr(phi, cpr_code, p, inv_peak):
    """Normalized CPR (peaks at 1)."""
    return _cpr_unnorm(phi, cpr_code, p) * inv_peak


def _cpr_peak(cpr_code, p):
    """Numerical peak of unnormalized CPR over phi in [0, pi]."""
    phi_grid = np.linspace(0.0, np.pi, 8001)   # finer for sharper peaks
    if cpr_code == 0:
        return 1.0
    elif cpr_code == 1:
        s = np.sin(phi_grid * 0.5)
        vals = np.sin(phi_grid) / np.sqrt(1.0 - p * s * s)
    elif cpr_code == 2:
        vals = np.zeros_like(phi_grid)
        for n in range(1, N_HARMONICS_LONG_BAL + 1):
            sign = 1.0 if (n % 2 == 1) else -1.0
            vals += sign / n * np.exp(-n * p) * np.sin(n * phi_grid)
    elif cpr_code == 3:
        s = np.sin(phi_grid * 0.5)
        c = np.cos(phi_grid * 0.5)
        s_clipped = np.clip(s, -0.99999, 0.99999)
        vals = c * np.arctanh(s_clipped)
    else:
        return 1.0
    return float(np.max(np.abs(vals)))


# ---------------------------------------------------------------------------
# Quasiparticle current F(v)
# ---------------------------------------------------------------------------
SOFT_SIGN_WIDTH = 0.05


@njit(cache=True, fastmath=True, inline='always')
def _F_qp(v, F_code, i_exc):
    """Dimensionless quasiparticle current.
    F_code = 0: linear F(v) = v
    F_code = 1: BTK F(v) = v + i_exc * tanh(v/w)
    """
    if F_code == 0:
        return v
    elif F_code == 1:
        return v + i_exc * np.tanh(v / SOFT_SIGN_WIDTH)
    else:
        return v


# ---------------------------------------------------------------------------
# Sweep builder
# ---------------------------------------------------------------------------
def _build_sweep(i_max, n_points):
    n_low    = max(2, n_points // 6)
    n_switch = n_points - 2 * (n_points // 6)
    n_high   = n_points // 6
    leg_up = np.concatenate([
        np.linspace(0.0, 0.5,    n_low,    endpoint=False),
        np.linspace(0.5, 1.3,    n_switch, endpoint=False),
        np.linspace(1.3, i_max,  n_high),
    ])
    leg_down = np.concatenate([leg_up[::-1], -leg_up[1:]])
    leg_up2  = -leg_up[::-1]
    return np.concatenate([leg_up, leg_down[1:], leg_up2[1:]])


# ---------------------------------------------------------------------------
# Single-junction sweeps (deterministic RK4, stochastic Euler-Maruyama)
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def _sweep_one_det(Q, sweep, t_transient, t_average, dt,
                   cpr_code, cpr_p, inv_peak, F_code, i_exc):
    M = sweep.shape[0]
    n_trans = max(1, int(round(t_transient / dt)))
    n_avg   = max(1, int(round(t_average   / dt)))
    inv_Q = 1.0 / Q
    half  = 0.5 * dt
    sixth = dt / 6.0
    TWO_PI = 2.0 * np.pi
    phi = 0.0
    phidot = 0.0
    v_trace = np.empty(M, dtype=np.float64)

    for k in range(M):
        i_b = sweep[k]
        for _ in range(n_trans):
            v1 = phidot * inv_Q
            k1p = phidot
            k1v = i_b - _cpr(phi, cpr_code, cpr_p, inv_peak) - _F_qp(v1, F_code, i_exc)

            phi2 = phi + half * k1p
            pd2  = phidot + half * k1v
            v2   = pd2 * inv_Q
            k2p = pd2
            k2v = i_b - _cpr(phi2, cpr_code, cpr_p, inv_peak) - _F_qp(v2, F_code, i_exc)

            phi3 = phi + half * k2p
            pd3  = phidot + half * k2v
            v3   = pd3 * inv_Q
            k3p = pd3
            k3v = i_b - _cpr(phi3, cpr_code, cpr_p, inv_peak) - _F_qp(v3, F_code, i_exc)

            phi4 = phi + dt * k3p
            pd4  = phidot + dt * k3v
            v4   = pd4 * inv_Q
            k4p = pd4
            k4v = i_b - _cpr(phi4, cpr_code, cpr_p, inv_peak) - _F_qp(v4, F_code, i_exc)

            phi    = phi    + sixth * (k1p + 2.0*k2p + 2.0*k3p + k4p)
            phidot = phidot + sixth * (k1v + 2.0*k2v + 2.0*k3v + k4v)

        sum_pd = 0.0
        for _ in range(n_avg):
            v1 = phidot * inv_Q
            k1p = phidot
            k1v = i_b - _cpr(phi, cpr_code, cpr_p, inv_peak) - _F_qp(v1, F_code, i_exc)

            phi2 = phi + half * k1p
            pd2  = phidot + half * k1v
            v2   = pd2 * inv_Q
            k2p = pd2
            k2v = i_b - _cpr(phi2, cpr_code, cpr_p, inv_peak) - _F_qp(v2, F_code, i_exc)

            phi3 = phi + half * k2p
            pd3  = phidot + half * k2v
            v3   = pd3 * inv_Q
            k3p = pd3
            k3v = i_b - _cpr(phi3, cpr_code, cpr_p, inv_peak) - _F_qp(v3, F_code, i_exc)

            phi4 = phi + dt * k3p
            pd4  = phidot + dt * k3v
            v4   = pd4 * inv_Q
            k4p = pd4
            k4v = i_b - _cpr(phi4, cpr_code, cpr_p, inv_peak) - _F_qp(v4, F_code, i_exc)

            phi    = phi    + sixth * (k1p + 2.0*k2p + 2.0*k3p + k4p)
            phidot = phidot + sixth * (k1v + 2.0*k2v + 2.0*k3v + k4v)
            sum_pd += phidot

        v_trace[k] = (sum_pd / n_avg) * inv_Q
        phi = (phi + np.pi) % TWO_PI - np.pi
    return v_trace


@njit(cache=True, fastmath=True)
def _sweep_one_sde(Q, gamma_T, sweep, t_transient, t_average, dt,
                   cpr_code, cpr_p, inv_peak, F_code, i_exc, seed):
    np.random.seed(seed)
    M = sweep.shape[0]
    n_trans = max(1, int(round(t_transient / dt)))
    n_avg   = max(1, int(round(t_average   / dt)))
    inv_Q = 1.0 / Q
    TWO_PI = 2.0 * np.pi
    noise_amp = np.sqrt(2.0 * gamma_T * inv_Q * dt)

    phi = 0.0
    phidot = 0.0
    v_trace = np.empty(M, dtype=np.float64)

    for k in range(M):
        i_b = sweep[k]
        for _ in range(n_trans):
            v = phidot * inv_Q
            drift_phi    = phidot
            drift_phidot = i_b - _cpr(phi, cpr_code, cpr_p, inv_peak) - _F_qp(v, F_code, i_exc)
            phi    = phi    + drift_phi    * dt
            phidot = phidot + drift_phidot * dt + noise_amp * np.random.randn()

        sum_pd = 0.0
        for _ in range(n_avg):
            v = phidot * inv_Q
            drift_phi    = phidot
            drift_phidot = i_b - _cpr(phi, cpr_code, cpr_p, inv_peak) - _F_qp(v, F_code, i_exc)
            phi    = phi    + drift_phi    * dt
            phidot = phidot + drift_phidot * dt + noise_amp * np.random.randn()
            sum_pd += phidot

        v_trace[k] = (sum_pd / n_avg) * inv_Q
        phi = (phi + np.pi) % TWO_PI - np.pi
    return v_trace


# ---------------------------------------------------------------------------
# Parallel batch kernel
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True, parallel=True)
def _sweep_batch(Qs, gamma_Ts, seeds, sweep, t_transient, t_average, dt,
                 cpr_codes, cpr_ps, inv_peaks, F_codes, i_excs):
    N = Qs.shape[0]
    M = sweep.shape[0]
    V = np.empty((N, M), dtype=np.float64)
    for n in prange(N):
        if gamma_Ts[n] == 0.0:
            V[n, :] = _sweep_one_det(
                Qs[n], sweep, t_transient, t_average, dt,
                cpr_codes[n], cpr_ps[n], inv_peaks[n],
                F_codes[n], i_excs[n])
        else:
            V[n, :] = _sweep_one_sde(
                Qs[n], gamma_Ts[n], sweep, t_transient, t_average, dt,
                cpr_codes[n], cpr_ps[n], inv_peaks[n],
                F_codes[n], i_excs[n], seeds[n])
    return V


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
_CPR_NAME_TO_CODE = {'sin': 0, 'ko1': 1, 'longbal': 2, 'ko2': 3}
_F_NAME_TO_CODE   = {'linear': 0, 'btk': 1}


def simulate_iv(Q, i_max=6.0, n_points=201,
                t_transient=200.0, t_average=200.0, dt=None,
                gamma_T=0.0, cpr='sin', cpr_p=0.0,
                F_kind='linear', i_exc=0.0,
                seed=None):
    """Simulate hysteretic IV sweeps for one or many junctions.

    Parameters
    ----------
    Q, gamma_T, cpr_p, i_exc : scalar or 1-D array of length N
        Per-junction parameters.
    cpr : 'sin' | 'ko1' | 'longbal' | 'ko2' | int | array of ints
        CPR selection. Names map: sin=0, ko1=1, longbal=2, ko2=3.
        cpr_p is the shape parameter (tau for ko1, L/xi for longbal,
        unused for sin/ko2).
    F_kind : 'linear' | 'btk' | int | array of ints
        Dissipative branch selection. linear=0, btk=1. i_exc is used
        only for btk.
    dt : float, optional
        Step size. Defaults to 0.05 (deterministic) or 0.02 (stochastic).
    seed : int, optional
        Master RNG seed.

    Returns
    -------
    sweep : (M,) dimensionless bias array
    v     : (M,) if all inputs scalar, else (N, M)
    """
    Q_in    = np.asarray(Q,       dtype=np.float64)
    gT_in   = np.asarray(gamma_T, dtype=np.float64)
    cpr_p_in = np.asarray(cpr_p,  dtype=np.float64)
    iexc_in = np.asarray(i_exc,   dtype=np.float64)

    scalar_input = (Q_in.ndim == 0 and gT_in.ndim == 0 and
                    cpr_p_in.ndim == 0 and iexc_in.ndim == 0 and
                    (isinstance(cpr, str) or np.ndim(cpr) == 0) and
                    (isinstance(F_kind, str) or np.ndim(F_kind) == 0))

    Q_arr, gT_arr, cprp_arr, iexc_arr = np.broadcast_arrays(
        np.atleast_1d(Q_in), np.atleast_1d(gT_in),
        np.atleast_1d(cpr_p_in), np.atleast_1d(iexc_in))
    Q_arr    = np.ascontiguousarray(Q_arr)
    gT_arr   = np.ascontiguousarray(gT_arr)
    cprp_arr = np.ascontiguousarray(cprp_arr)
    iexc_arr = np.ascontiguousarray(iexc_arr)
    N = Q_arr.size

    # Resolve cpr codes
    if isinstance(cpr, str):
        cpr_arr = np.full(N, _CPR_NAME_TO_CODE[cpr.lower()], dtype=np.int64)
    else:
        cpr_arr = np.broadcast_to(np.asarray(cpr, dtype=np.int64), (N,))
        cpr_arr = np.ascontiguousarray(cpr_arr)

    # Resolve F codes
    if isinstance(F_kind, str):
        F_arr = np.full(N, _F_NAME_TO_CODE[F_kind.lower()], dtype=np.int64)
    else:
        F_arr = np.broadcast_to(np.asarray(F_kind, dtype=np.int64), (N,))
        F_arr = np.ascontiguousarray(F_arr)

    # Compute peak normalization per junction
    inv_peaks = np.empty(N, dtype=np.float64)
    for n in range(N):
        peak = _cpr_peak(int(cpr_arr[n]), float(cprp_arr[n]))
        inv_peaks[n] = 1.0 / peak if peak > 0 else 1.0
    inv_peaks = np.ascontiguousarray(inv_peaks)

    if dt is None:
        dt = 0.05 if np.all(gT_arr == 0.0) else 0.02

    sweep = _build_sweep(i_max, n_points)

    rng = np.random.default_rng(seed)
    seeds = rng.integers(low=1, high=2**31 - 1, size=N, dtype=np.int64)

    V = _sweep_batch(Q_arr, gT_arr, seeds, sweep,
                     t_transient, t_average, dt,
                     cpr_arr, cprp_arr, inv_peaks,
                     F_arr, iexc_arr)

    if scalar_input:
        return sweep, V[0]
    return sweep, V


if __name__ == "__main__":
    import time
    print("warming up JIT (first run triggers compilation)...")
    t0 = time.time()
    # Trigger compilation of all CPR types and both F kinds
    for cpr_name, p_val in [('sin', 0.0), ('ko1', 0.7),
                             ('longbal', 1.5), ('ko2', 0.0)]:
        for F_name, ie in [('linear', 0.0), ('btk', 0.5)]:
            _ = simulate_iv(Q=1.0, gamma_T=1e-3,
                            cpr=cpr_name, cpr_p=p_val,
                            F_kind=F_name, i_exc=ie,
                            n_points=5, t_transient=1.0, t_average=1.0, seed=0)
    print(f"  warmup: {time.time()-t0:.1f}s")

    # Quick batch-throughput test
    print("\nThroughput test (N=1000, 4 CPRs mixed, BTK on for KO):")
    rng = np.random.default_rng(0)
    N = 1000
    Qs   = rng.uniform(0.3, 5.0, size=N)
    gTs  = 10 ** rng.uniform(-4, -1.5, size=N)
    # mix all four CPRs
    cprs = rng.integers(0, 4, size=N)
    cprps = np.where(cprs == 1, rng.uniform(0.5, 0.95, size=N),
              np.where(cprs == 2, rng.uniform(0.5, 3.0, size=N),
                       0.0))
    F_kinds = np.where((cprs == 1) | (cprs == 3), 1, 0).astype(np.int64)
    iexcs = np.where(F_kinds == 1, rng.uniform(0.3, 0.7, size=N), 0.0)

    t0 = time.time()
    sweep, V = simulate_iv(Q=Qs, gamma_T=gTs,
                            cpr=cprs, cpr_p=cprps,
                            F_kind=F_kinds, i_exc=iexcs,
                            n_points=201,
                            t_transient=150.0, t_average=150.0, seed=42)
    elapsed = time.time() - t0
    print(f"  N={N}  elapsed={elapsed:.2f}s  ({1000*elapsed/N:.2f} ms/trace)")