import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import logging

# Log the Matplotlib version
print(f"Matplotlib version: {matplotlib.__version__}")
logging.basicConfig(
    filename='simulation_log.txt',
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
lambda_ = 200.0
eta = 400.0
G_grav = 50.0
charge_1 = 1.0
charge_2 = -1.0
alpha_em = 1/137
num_steps = 50
plot_interval = 5
erasure_time = 0.050

# Log simulation parameters
logging.info("Simulation Parameters:")
logging.info(f"M = {M}, J = {J}, L = {L}, X = {X}")
logging.info(f"dt = {dt}, beta = {beta}, hbar = {hbar}, m_xi = {m_xi}")
logging.info(f"kappa = {kappa}, lambda = {lambda_}, eta = {eta}, G_grav = {G_grav}, alpha_em = {alpha_em}, num_steps = {num_steps}, plot_interval = {plot_interval}, erasure_time = {erasure_time}")

# Internal space grid (2D for pre-erasure, 1D for post-erasure)
xi1 = np.linspace(-L, L, M, endpoint=False)
xi2 = np.linspace(-L, L, M, endpoint=False)
XI1, XI2 = np.meshgrid(xi1, xi2)

# Emergent space grid (2D)
x = np.linspace(-X, X, J, endpoint=False)
y = np.linspace(-X, X, J, endpoint=False)
X_grid, Y_grid = np.meshgrid(x, y)

# Initialize state function with entangled wavepackets
sigma = 1.0
xi1_1, xi1_2 = -4.0, 4.0
xi2_1, xi2_2 = -4.0, 4.0
k0 = 10.0
A = 1.0 / np.sqrt(2 * np.pi * sigma**2)

# Particle wavepackets
psi1 = np.exp(-(XI1 - xi1_1)**2 / (2 * sigma**2)) * np.exp(1j * k0 * XI1)
psi2 = np.exp(-(XI1 - xi1_2)**2 / (2 * sigma**2)) * np.exp(1j * k0 * XI1)

# Detector wavepackets
phi1 = np.exp(-(XI2 - xi2_1)**2 / (2 * sigma**2))
phi2 = np.exp(-(XI2 - xi2_2)**2 / (2 * sigma**2))

# Entangled initial state
psi = A * (psi1 * phi1 + psi2 * phi2)
psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx**2)

logging.info("Initial wavepacket centers: xi1_1 = %.2f, xi1_2 = %.2f, initial separation = %.2f", xi1_1, xi1_2, xi1_2 - xi1_1)
logging.info("Detector centers: xi2_1 = %.2f, xi2_2 = %.2f", xi2_1, xi2_2)
logging.info(f"sigma = {sigma}, k0 = {k0}")

# Plot initial internal state
plt.figure()
plt.contourf(XI1, XI2, np.abs(psi)**2, cmap='viridis')
plt.colorbar(label="Probability Density |ψ(xi1,xi2,0)|²")
plt.xlabel("Internal Space (xi1)")
plt.ylabel("Internal Space (xi2)")
plt.title("Initial State in Internal Space")
plt.savefig("initial_internal_state.png")
plt.close()

# Potential
V = kappa * (XI1**2 + XI2**2)

# Precompute kinetic term
kx1 = 2 * np.pi * np.fft.fftfreq(M, dx)
kx2 = 2 * np.pi * np.fft.fftfreq(M, dx)
KX1, KX2 = np.meshgrid(kx1, kx2)
kinetic = hbar**2 * (KX1**2 + KX2**2) / (2 * m_xi)

# Non-local kernel for 2D
K = 1 / (np.sqrt(XI1**2 + XI2**2) + 1e-6)
K_fft = np.fft.fft2(K)

# Non-local kernel for 1D
K_1d = 1 / (np.abs(xi1) + 1e-6)
K_fft_1d = np.fft.fft(K_1d)

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
fig_internal, axes_internal = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4), sharex=True, sharey=False)
fig_emergent, axes_emergent = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4), sharex=True, sharey=True)
if num_plots == 1:
    axes_internal = [axes_internal]
    axes_emergent = [axes_emergent]

# Flag to track whether erasure has been applied
erasure_applied = False
psi_1d = None

# Time evolution
for step in range(num_steps):
    t = step * dt

    # Compute wavepacket centers
    if not erasure_applied:
        prob = np.abs(psi)**2
        prob_xi1 = np.sum(prob, axis=0) * dx
        prob_xi1 /= np.sum(prob_xi1) * dx
    else:
        prob = np.abs(psi_1d)**2
        prob_xi1 = prob / np.sum(prob * dx)

    prob_left = prob_xi1.copy()
    prob_left[M//2:] = 0
    prob_left /= np.sum(prob_left) * dx
    center_left = np.sum(xi1 * prob_left) * dx

    prob_right = prob_xi1.copy()
    prob_right[:M//2] = 0
    prob_right /= np.sum(prob_right) * dx
    center_right = np.sum(xi1 * prob_right) * dx

    separation = center_right - center_left

    # Gravitational potential
    V_grav = np.zeros_like(XI1)
    r = np.abs(XI1 - center_right)
    r[r < 1e-6] = 1e-6
    V_grav += -G_grav * prob_left[int(M/2)] / r
    r = np.abs(XI1 - center_left)
    r[r < 1e-6] = 1e-6
    V_grav += -G_grav * prob_right[int(M/2)] / r

    if not erasure_applied:
        # Compute non-local term G[ψ]
        psi_density = np.abs(psi)**2
        psi_density_fft = np.fft.fft2(psi_density)
        G = np.fft.ifft2(K_fft * psi_density_fft).real

        # Total potential
        V_total = V + lambda_ * np.abs(psi)**2 + eta * G + V_grav

        # Apply potential term
        psi = np.exp(-1j * dt * V_total / hbar) * psi

        # Apply kinetic term
        psi_fft = np.fft.fft2(psi)
        psi_fft = np.exp(-1j * dt * kinetic / hbar) * psi_fft
        psi = np.fft.ifft2(psi_fft)

        # Renormalize
        psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx**2)
    else:
        # 1D evolution after erasure
        V_grav_1d = np.zeros_like(xi1)
        r = np.abs(xi1 - center_right)
        r[r < 1e-6] = 1e-6
        V_grav_1d += -G_grav * prob_left[int(M/2)] / r
        r = np.abs(xi1 - center_left)
        r[r < 1e-6] = 1e-6
        V_grav_1d += -G_grav * prob_right[int(M/2)] / r

        # 1D non-local term
        psi_density_1d = np.abs(psi_1d)**2
        psi_density_fft_1d = np.fft.fft(psi_density_1d)
        G_1d = np.fft.ifft(K_fft_1d * psi_density_fft_1d).real

        V_total_1d = kappa * xi1**2 + lambda_ * np.abs(psi_1d)**2 + eta * G_1d + V_grav_1d

        psi_1d = np.exp(-1j * dt * V_total_1d / hbar) * psi_1d

        kx1_1d = 2 * np.pi * np.fft.fftfreq(M, dx)
        kinetic_1d = hbar**2 * kx1_1d**2 / (2 * m_xi)
        psi_fft_1d = np.fft.fft(psi_1d)
        psi_fft_1d = np.exp(-1j * dt * kinetic_1d / hbar) * psi_fft_1d
        psi_1d = np.fft.ifft(psi_fft_1d)

        psi_1d /= np.sqrt(np.sum(np.abs(psi_1d)**2) * dx)

    # Log wavepacket centers
    logging.info(f"Step {step}, Time {t:.3f}: Wavepacket centers: left = {center_left:.2f}, right = {center_right:.2f}, separation = {separation:.2f}")

    # Apply quantum erasure at the specified time
    if t >= erasure_time and not erasure_applied:
        logging.info(f"Applying quantum erasure at time t = {t:.3f}")
        # Compute the projection components separately to preserve phase
        erasure_state = (phi1 + phi2) / np.sqrt(2)
        erasure_state /= np.sqrt(np.sum(np.abs(erasure_state)**2) * dx)

        # Project each component of the entangled state
        psi_1d_left = np.sum(psi1 * phi1 * erasure_state.conj(), axis=1) * dx
        psi_1d_right = np.sum(psi2 * phi2 * erasure_state.conj(), axis=1) * dx
        psi_1d = psi_1d_left + psi_1d_right
        psi_1d /= np.sqrt(np.sum(np.abs(psi_1d)**2) * dx)
        erasure_applied = True

    # FFT-accelerated ξ → x transform and electromagnetic effects
    if t in [0.000, 0.004, 0.008] or step % plot_interval == 0:
        if not erasure_applied:
            Psi_fft = np.fft.fft2(psi) * dx**2
            Psi = np.zeros((J, J), dtype=complex)
            for jx in range(J):
                for jy in range(J):
                    kx_shift = beta * k0 - KX[jx, jy]
                    ky_shift = beta * k0 - KY[jx, jy]
                    Psi[jx, jy] = np.sum(Psi_fft * np.exp(-1j * (kx_shift * XI1 + ky_shift * XI2))) * (kx[1] - kx[0]) * (ky[1] - ky[0]) / (2 * np.pi)**2
        else:
            Psi = np.zeros((J, J), dtype=complex)
            for jx in range(J):
                for jy in range(J):
                    # Compute contributions from both wavepackets with phase difference
                    kx_shift = beta * k0 - KX[jx, jy]
                    # Left wavepacket
                    psi_left = np.exp(-(xi1 - center_left)**2 / (2 * sigma**2)) * np.exp(1j * k0 * xi1)
                    psi_left_fft = np.fft.fft(psi_left) * dx
                    Psi_left = np.sum(psi_left_fft * np.exp(-1j * kx_shift * xi1)) * (kx[1] - kx[0]) / (2 * np.pi)
                    # Right wavepacket
                    psi_right = np.exp(-(xi1 - center_right)**2 / (2 * sigma**2)) * np.exp(1j * k0 * xi1)
                    psi_right_fft = np.fft.fft(psi_right) * dx
                    Psi_right = np.sum(psi_right_fft * np.exp(-1j * kx_shift * xi1)) * (kx[1] - kx[0]) / (2 * np.pi)
                    Psi[jx, jy] = Psi_left + Psi_right

        phi = np.zeros((J, J))
        for jx in range(J):
            for jy in range(J):
                r1 = np.sqrt((x[jx] - center_left)**2 + y[jy]**2 + 1e-6)
                r2 = np.sqrt((x[jx] - center_right)**2 + y[jy]**2 + 1e-6)
                phi[jx, jy] = alpha_em * (charge_1 / r1 + charge_2 / r2)

        Psi *= np.exp(-1j * phi / hbar)

        prob_emergent = np.abs(Psi)**2

        y_center_idx = J // 2
        prob_slice = prob_emergent[y_center_idx, :]
        fft_slice = np.fft.fft(prob_slice)
        freqs = np.fft.fftfreq(J, dx_emergent)
        fft_magnitude = np.abs(fft_slice)
        fft_magnitude[0] = 0
        threshold = 0.2 * np.max(fft_magnitude)
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
            if not erasure_applied:
                c_internal = ax_internal.contourf(XI1, XI2, prob, cmap=colormaps[plot_idx])
                ax_internal.set_ylabel("Internal Space (xi2)")
                fig_internal.colorbar(c_internal, ax=ax_internal, label="|ψ(xi1,xi2,t)|²")
            else:
                ax_internal.plot(xi1, prob, color='orange')
                ax_internal.set_ylim(0, np.max(prob) * 1.1)
                ax_internal.set_ylabel("Probability Density |ψ(xi1,t)|²")
            ax_internal.set_xlabel("Internal Space (xi1)")
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
fig_internal.savefig("internal_states.png")
fig_emergent.savefig("emergent_states.png")
plt.close(fig_internal)
plt.close(fig_emergent)

logging.info("Simulation complete. Plots saved as 'internal_states.png' and 'emergent_states.png'.")