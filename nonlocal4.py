"""
Non-Local Interference Simulation in the Point Universe Model (PUM) and Spacetime Superfluid Hypothesis (SSH)

This simulation explores interference patterns in emergent spacetime (x, y) arising from two wavepackets in a 2D internal space (ξ1, ξ2), incorporating non-local interactions via a Coulomb-like kernel. It is part of a series of theoretical physics papers by Eric Albers

Related Papers:
- Point Universe Model (PUM, papers/pointmodel.pdf): This simulation tests PUM's double-slit interference prediction (Section 9.1), where internal wavepackets produce emergent interference patterns, and non-local interactions reflect quantum entanglement (Section 3.8.2). The non-local term η G[ψ] uses a Coulomb-like kernel, mirroring PUM's framework for long-range correlations.
- Spacetime Superfluid Hypothesis (SSH, papers/pointmodelssh.pdf): The non-linear Schrödinger equation (NLSE) with non-local interactions aligns with SSH's superfluid dynamics (Section 3), where particles interact via the superfluid medium (Section 4.3).
- Tired Light in SSH (papers/tiredlight.pdf): The non-local framework could be extended to include dissipative terms, testing tired light's energy loss mechanism for cosmological redshift (Section 5.1).
- Black Holes as Superfluid Vortices (papers/blackholes_ssh.pdf): The 2D internal space could be adapted to model vortices, testing interference near black hole horizons (Section 4.5).

Simulation Details:
- Initializes two Gaussian wavepackets at ξ1 = ±4.0 with a Gaussian envelope in ξ2, evolving under the NLSE with non-local interactions (η = 400).
- Maps the internal wavefunction to emergent spacetime via Fourier transform, producing 2D interference patterns.
- Logs wavepacket separation and fringe counts, visualizing the impact of non-local effects on interference.
- Outputs: internal_states.png (internal probability density), emergent_states.png (emergent interference patterns), simulation_log.txt (fringe counts and dynamics).

Usage:
- Run the script to generate interference patterns: `python nonlocal4.py`.
- Outputs are saved as PNG plots and a log file.
- Modify parameters (e.g., η, λ, κ) to explore different dynamics, or add dissipative terms to test tired light effects.

"""


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import logging

# Log the Matplotlib version
print(f"Matplotlib version: {matplotlib.__version__}")
logging.basicConfig(
    filename='nonlocal_simulation_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info(f"Matplotlib version: {matplotlib.__version__}")

# Parameters
M = 128
L = 20.0
dx = 2 * L / M
J = 256
X = 20.0
dx_emergent = 2 * X / J
dt = 0.002
beta = 1.0
hbar = 1.0
m_xi = 1.0
kappa = 0.0
lambda_ = 200.0  # Moderated for cleaner dynamics
eta = 400.0  # Moderated for cleaner non-local effects
num_steps = 50
plot_interval = 5

# Log simulation parameters
logging.info("Simulation Parameters:")
logging.info(f"M = {M}, J = {J}, L = {L}, X = {X}")
logging.info(f"dt = {dt}, beta = {beta}, hbar = {hbar}, m_xi = {m_xi}")
logging.info(f"kappa = {kappa}, lambda = {lambda_}, eta = {eta}, num_steps = {num_steps}, plot_interval = {plot_interval}")

# Internal space grid (2D)
xi1 = np.linspace(-L, L, M, endpoint=False)
xi2 = np.linspace(-L, L, M, endpoint=False)
XI1, XI2 = np.meshgrid(xi1, xi2)

# Emergent space grid (2D)
x = np.linspace(-X, X, J, endpoint=False)
y = np.linspace(-X, X, J, endpoint=False)
X_grid, Y_grid = np.meshgrid(x, y)

# Initialize state function
sigma = 1.0
xi1_1, xi1_2 = -4.0, 4.0
xi2_center = 0.0
k0 = 10.0
A = 1.0 / np.sqrt(2 * np.pi * sigma**2)
psi = A * np.exp(-(XI2 - xi2_center)**2 / (2 * sigma**2)) * (
    np.exp(-(XI1 - xi1_1)**2 / (2 * sigma**2)) * np.exp(1j * k0 * XI1) +
    np.exp(-(XI1 - xi1_2)**2 / (2 * sigma**2)) * np.exp(1j * k0 * XI1)
)
psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx**2)

logging.info("Initial wavepacket centers: xi1_1 = %.2f, xi1_2 = %.2f, initial separation = %.2f", xi1_1, xi1_2, xi1_2 - xi1_1)
logging.info(f"sigma = {sigma}, k0 = {k0}")

# Plot initial internal state
plt.figure()
plt.contourf(XI1, XI2, np.abs(psi)**2, cmap='viridis')
plt.colorbar(label="Probability Density |ψ(xi1,xi2,0)|²")
plt.xlabel("Internal Space (xi1)")
plt.ylabel("Internal Space (xi2)")
plt.title("Initial State in Internal Space")
plt.savefig("nonlocal_initial_internal_state.png")
plt.close()

# Potential
V = kappa * (XI1**2 + XI2**2)

# Precompute kinetic term
kx1 = 2 * np.pi * np.fft.fftfreq(M, dx)
kx2 = 2 * np.pi * np.fft.fftfreq(M, dx)
KX1, KX2 = np.meshgrid(kx1, kx2)
kinetic = hbar**2 * (KX1**2 + KX2**2) / (2 * m_xi)

# Define the non-local kernel (Coulomb-like)
K = 1 / (np.sqrt(XI1**2 + XI2**2) + 1e-6)
K_fft = np.fft.fft2(K)

# Precompute for FFT-based ξ → x transform
kx = 2 * np.pi * np.fft.fftfreq(J, dx_emergent)
ky = 2 * np.pi * np.fft.fftfreq(J, dx_emergent)
KX, KY = np.meshgrid(kx, ky)

# Set up colormaps
num_plots = num_steps // plot_interval
colormaps = [cm.get_cmap('viridis'), cm.get_cmap('plasma'), cm.get_cmap('inferno'), cm.get_cmap('magma'), cm.get_cmap('cividis')]
if num_plots > len(colormaps):
    colormaps = colormaps * (num_plots // len(colormaps) + 1)

# Create figures
fig_internal, axes_internal = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4), sharex=True, sharey=True)
fig_emergent, axes_emergent = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4), sharex=True, sharey=True)
if num_plots == 1:
    axes_internal = [axes_internal]
    axes_emergent = [axes_emergent]

# Time evolution
for step in range(num_steps):
    t = step * dt

    # Compute non-local term G[ψ]
    psi_density = np.abs(psi)**2
    psi_density_fft = np.fft.fft2(psi_density)
    G = np.fft.ifft2(K_fft * psi_density_fft).real

    # Apply potential, local, and non-local terms
    V_total = V + lambda_ * np.abs(psi)**2 + eta * G
    psi = np.exp(-1j * dt * V_total / hbar) * psi

    # Apply kinetic term
    psi_fft = np.fft.fft2(psi)
    psi_fft = np.exp(-1j * dt * kinetic / hbar) * psi_fft
    psi = np.fft.ifft2(psi_fft)

    # Renormalize
    psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx**2)

    # Compute wavepacket centers using expectation values
    prob = np.abs(psi)**2
    prob_xi1 = np.sum(prob, axis=0) * dx
    prob_xi1 /= np.sum(prob_xi1) * dx

    prob_left = prob_xi1.copy()
    prob_left[M//2:] = 0
    prob_left /= np.sum(prob_left) * dx
    center_left = np.sum(xi1 * prob_left) * dx

    prob_right = prob_xi1.copy()
    prob_right[:M//2] = 0
    prob_right /= np.sum(prob_right) * dx
    center_right = np.sum(xi1 * prob_right) * dx

    separation = center_right - center_left
    logging.info(f"Step {step}, Time {t:.3f}: Wavepacket centers: left = {center_left:.2f}, right = {center_right:.2f}, separation = {separation:.2f}")

    # FFT-accelerated ξ → x transform
    if t in [0.000, 0.004, 0.008] or step % plot_interval == 0:
        # Compute Ψ(x, y, t) using FFT
        Psi_fft = np.fft.fft2(psi) * dx**2
        Psi = np.zeros((J, J), dtype=complex)
        for jx in range(J):
            for jy in range(J):
                kx_shift = beta * k0 - KX[jx, jy]
                ky_shift = beta * k0 - KY[jx, jy]
                Psi[jx, jy] = np.sum(Psi_fft * np.exp(-1j * (kx_shift * XI1 + ky_shift * XI2))) * (kx[1] - kx[0]) * (ky[1] - ky[0]) / (2 * np.pi)**2
        prob_emergent = np.abs(Psi)**2

        # Fourier-based fringe counting
        y_center_idx = J // 2
        prob_slice = prob_emergent[y_center_idx, :]
        fft_slice = np.fft.fft(prob_slice)
        freqs = np.fft.fftfreq(J, dx_emergent)
        fft_magnitude = np.abs(fft_slice)
        fft_magnitude[0] = 0  # Ignore DC component
        threshold = 0.1 * np.max(fft_magnitude)
        significant_peaks = fft_magnitude > threshold
        num_fringes = np.sum(significant_peaks[:J//2])
        dominant_freq_idx = np.argmax(fft_magnitude[:J//2])
        dominant_freq = freqs[dominant_freq_idx]
        fringe_spacing = 1 / dominant_freq if dominant_freq != 0 else float('inf')

        logging.info(f"Step {step}, Time {t:.3f}: Number of fringes (Fourier-based) = {num_fringes}")
        logging.info(f"Step {step}, Time {t:.3f}: Dominant frequency (emergent space) = {dominant_freq:.2f} cycles/unit length")
        logging.info(f"Step {step}, Time {t:.3f}: Fringe spacing (emergent space) = {fringe_spacing:.2f} units")
        logging.info(f"Step {step}, Time {t:.3f}: Probability slice data (first 50 values) = {prob_slice[:50].tolist()}")

        if step % plot_interval == 0:
            plot_idx = step // plot_interval
            ax_internal = axes_internal[plot_idx]
            c_internal = ax_internal.contourf(XI1, XI2, prob, cmap=colormaps[plot_idx])
            fig_internal.colorbar(c_internal, ax=ax_internal, label="|ψ(xi1,xi2,t)|²")
            ax_internal.set_xlabel("Internal Space (xi1)")
            ax_internal.set_ylabel("Internal Space (xi2)")
            ax_internal.set_title(f"Time: {t:.3f}\nSep: {separation:.2f}")

            ax_emergent = axes_emergent[plot_idx]
            c_emergent = ax_emergent.contourf(X_grid, Y_grid, prob_emergent.T, cmap=colormaps[plot_idx])
            fig_emergent.colorbar(c_emergent, ax=ax_emergent, label="|Ψ(x,y,t)|²")
            ax_emergent.set_xlabel("Emergent Space (x)")
            ax_emergent.set_ylabel("Emergent Space (y)")
            ax_emergent.set_title(f"Time: {t:.3f}\nFringes: {num_fringes}")

# Save plots
fig_internal.tight_layout()
fig_emergent.tight_layout()
fig_internal.savefig("nonlocal_internal_states.png")
fig_emergent.savefig("nonlocal_emergent_states.png")
plt.close(fig_internal)
plt.close(fig_emergent)

logging.info("Simulation complete. Plots saved as 'nonlocal_internal_states.png' and 'nonlocal_emergent_states.png'.")