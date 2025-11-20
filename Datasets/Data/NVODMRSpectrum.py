import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from scipy.linalg import eigh
from textwrap import dedent


def spin_one_matrices():
    Sx = 0.5 * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=complex)
    Sy = 0.5 * np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], dtype=complex)
    Sz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype=complex)
    return Sx, Sy, Sz


def make_cs(nv):
    v1 = np.asarray(nv, dtype=float)
    z = np.array([0.0, 0.0, 1.0], dtype=float)
    v2 = np.cross(v1, z)
    if np.linalg.norm(v2) < 1e-12:
        v2 = np.array([1.0, 0.0, 0.0])
    v3 = np.cross(v1, v2)
    return v1, v2, v3


def proj_B_to_NV(B, nv):
    v1, v2, v3 = make_cs(nv)
    vx, vy, vz = (
        v2 / np.linalg.norm(v2),
        v3 / np.linalg.norm(v3),
        v1 / np.linalg.norm(v1),
    )
    return float(np.dot(B, vx)), float(np.dot(B, vy)), float(np.dot(B, vz))


def kron(a, b):
    return np.kron(a, b)


def build_hamiltonian(B_vec, nv_axis, params):
    Dz, P, gamma_e, gamma_n, A1, Az = params
    Sx, Sy, Sz = spin_one_matrices()
    I3 = np.eye(3, dtype=complex)
    Bx, By, Bz = proj_B_to_NV(np.asarray(B_vec, float), np.asarray(nv_axis, float))
    Hzps = Dz * kron(Sz @ Sz, I3)
    Hqp = P * kron(I3, Sz @ Sz)
    HzeemanE = gamma_e * kron(Bx * Sx + By * Sy + Bz * Sz, I3)
    HzeemanN = -gamma_n * kron(I3, Bx * Sx + By * Sy + Bz * Sz)
    Hhf = A1 * (kron(Sx, Sx) + kron(Sy, Sy)) + Az * kron(Sz, Sz)
    return Hzps + Hqp + HzeemanE + HzeemanN + Hhf


def energies_MHz(B_vec, nv_axis, params):
    evals, _ = eigh(build_hamiltonian(B_vec, nv_axis, params))
    return np.sort(np.real(evals) / (2 * pi))


def lorentzian(x, center, width, amp):
    return 1.0 - amp * (width**2) / ((x - center) ** 2 + width**2)


def gen_spectrum_nv(B0, theta_deg, phi_deg, width, amp, freq_grid):
    NVs100 = np.array(
        [[1, 1, 1], [-1, -1, 1], [1, -1, -1], [-1, 1, -1]], dtype=float
    ) / np.sqrt(3.0)
    # params (2π·MHz)
    Dz = 2 * pi * 2870.0
    P = -2 * pi * 4.9457
    gamma_e = 2 * pi * 2.803
    gamma_n = 2 * pi * 3.077e-4
    A1 = -2 * pi * 2.62
    Az = 2 * pi * 2.2
    params = (Dz, P, gamma_e, gamma_n, A1, Az)

    th, ph = np.deg2rad(theta_deg), np.deg2rad(phi_deg)
    B_lab = np.array(
        [B0 * np.sin(th) * np.cos(ph), B0 * np.sin(th) * np.sin(ph), B0 * np.cos(th)],
        dtype=float,
    )

    centers = []
    for nv in NVs100:
        ev = energies_MHz(B_lab, nv, params)
        groups = [ev[i : i + 3] for i in range(0, 9, 3)]
        # pick ms=0 group by mean closest to 0
        idx_mid = int(np.argmin([abs(g.mean()) for g in groups]))
        idx_other = [i for i in range(3) if i != idx_mid]
        g_mid = groups[idx_mid]
        g_a, g_b = groups[idx_other[0]], groups[idx_other[1]]
        for i in range(3):
            centers.append(abs(g_a[i] - g_mid[i]))
            centers.append(abs(g_b[i] - g_mid[i]))
    centers = np.array(centers, float)

    # Simulate random strain by removing some centers
    # centers = np.random.choice(centers, size=8, replace=False)

    y = np.zeros_like(freq_grid, float)
    for c in centers:
        y += lorentzian(freq_grid, c, width, amp)
    y /= len(centers) if len(centers) else 1.0
    return y, centers
