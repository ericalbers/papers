'''
The interference.py script simulates the evolution of a wavefunction in an internal space (ξ\xi\xi
) and maps it to an emergent spacetime ((x)) using a Fourier transform, focusing on the interference pattern created by two Gaussian wavepackets—mimicking a double-slit-like setup. This is a simplified version of your previous quantum eraser simulation, focusing on free evolution (no external potential or dissipation) to highlight interference in the emergent space. Here’s a breakdown of what the code does:
Setup and Parameters:
Internal Space: A 1D grid with M=4096M = 4096M = 4096
 points, spanning ξ∈[−L,L]\xi \in [-L, L]\xi \in [-L, L]
 where L=20.0L = 20.0L = 20.0
, with spacing dx=2L/M=0.00977dx = 2L/M = 0.00977dx = 2L/M = 0.00977
.

Emergent Space: A 1D grid with J=4096J = 4096J = 4096
 points, spanning x∈[−X,X]x \in [-X, X]x \in [-X, X]
 where X=20.0X = 20.0X = 20.0
, with spacing dxemergent=2X/J=0.00977dx_{\text{emergent}} = 2X/J = 0.00977dx_{\text{emergent}} = 2X/J = 0.00977
.

Time: Runs for numsteps=100num_steps = 100num_steps = 100
 steps with dt=0.001dt = 0.001dt = 0.001
, totaling t=0.1t = 0.1t = 0.1
, plotting every plotinterval=10plot_interval = 10plot_interval = 10
 steps (t=0,0.010,…,0.090t = 0, 0.010, \ldots, 0.090t = 0, 0.010, \ldots, 0.090
).

Physical Parameters: ℏ=1\hbar = 1\hbar = 1
, mξ=1m_\xi = 1m_\xi = 1
, κ=0\kappa = 0\kappa = 0
 (no potential), β=1\beta = 1\beta = 1
 (mapping scale), k0=5.0k_0 = 5.0k_0 = 5.0
 (wavenumber), σ=0.5\sigma = 0.5\sigma = 0.5
 (wavepacket width).

Initial State:
The wavefunction ψ(ξ,0)\psi(\xi, 0)\psi(\xi, 0)
 is a superposition of two Gaussian wavepackets centered at ξ1=−4.0\xi_1 = -4.0\xi_1 = -4.0
 and ξ2=4.0\xi_2 = 4.0\xi_2 = 4.0
, each with a phase factor eik0ξe^{i k_0 \xi}e^{i k_0 \xi}
:
ψ(ξ,0)=A[e−(ξ−ξ1)2/(2σ2)eik0ξ+e−(ξ−ξ2)2/(2σ2)eik0ξ]\psi(\xi, 0) = A \left[ e^{-(\xi - \xi_1)^2 / (2 \sigma^2)} e^{i k_0 \xi} + e^{-(\xi - \xi_2)^2 / (2 \sigma^2)} e^{i k_0 \xi} \right]\psi(\xi, 0) = A \left[ e^{-(\xi - \xi_1)^2 / (2 \sigma^2)} e^{i k_0 \xi} + e^{-(\xi - \xi_2)^2 / (2 \sigma^2)} e^{i k_0 \xi} \right]
where A=1/2πσ2A = 1 / \sqrt{2 \pi \sigma^2}A = 1 / \sqrt{2 \pi \sigma^2}
 normalizes the amplitude, and the wavefunction is further normalized to ensure ∫∣ψ∣2dξ=1\int |\psi|^2 d\xi = 1\int |\psi|^2 d\xi = 1
.

This mimics a double-slit experiment in internal space, with the two Gaussians representing wavepackets passing through two slits, and k0=5.0k_0 = 5.0k_0 = 5.0
 setting the initial momentum (wavelength λ=2π/k0≈1.257\lambda = 2\pi/k_0 \approx 1.257\lambda = 2\pi/k_0 \approx 1.257
).

Time Evolution:
The wavefunction evolves using the split-operator method, a common technique for solving the Schrödinger equation:
Potential Term: With κ=0\kappa = 0\kappa = 0
, the potential V=κξ2=0V = \kappa \xi^2 = 0V = \kappa \xi^2 = 0
, so this step is a no-op (ψ→e−idtV/ℏψ=ψ\psi \to e^{-i dt V / \hbar} \psi = \psi\psi \to e^{-i dt V / \hbar} \psi = \psi
).

Kinetic Term: Applied in momentum space via FFT:
T=ℏ2kx22mξ,ψ→ifft(e−idtT/ℏfft(ψ))T = \frac{\hbar^2 k_x^2}{2 m_\xi}, \quad \psi \to \text{ifft}(e^{-i dt T / \hbar} \text{fft}(\psi))T = \frac{\hbar^2 k_x^2}{2 m_\xi}, \quad \psi \to \text{ifft}(e^{-i dt T / \hbar} \text{fft}(\psi))

The wavefunction is renormalized at each step to prevent numerical drift.

Mapping to Emergent Space:
At each plotting step, the internal wavefunction ψ(ξ,t)\psi(\xi, t)\psi(\xi, t)
 is Fourier-transformed to the emergent space:
Ψ(x,t)=∫ψ(ξ,t)e−ikxξdξ,kx=βk0−x\Psi(x, t) = \int \psi(\xi, t) e^{-i k_x \xi} d\xi, \quad k_x = \beta k_0 - x\Psi(x, t) = \int \psi(\xi, t) e^{-i k_x \xi} d\xi, \quad k_x = \beta k_0 - x
where β=1\beta = 1\beta = 1
 scales the mapping, and the integral is approximated as a sum:
Ψ(xj,t)=∑ξψ(ξ,t)e−i(βk0−xj)ξdx\Psi(x_j, t) = \sum_{\xi} \psi(\xi, t) e^{-i (\beta k_0 - x_j) \xi} dx\Psi(x_j, t) = \sum_{\xi} \psi(\xi, t) e^{-i (\beta k_0 - x_j) \xi} dx

The probability density ∣Ψ(x,t)∣2|\Psi(x, t)|^2|\Psi(x, t)|^2
 is computed and plotted.

Visualization:
Initial Plot: Plots ∣ψ(ξ,0)∣2|\psi(\xi, 0)|^2|\psi(\xi, 0)|^2
, showing two Gaussian peaks at ξ=±4\xi = \pm 4\xi = \pm 4
.

Emergent Space Plot: Plots ∣Ψ(x,t)∣2|\Psi(x, t)|^2|\Psi(x, t)|^2
 at each plotting step (t=0,0.010,…,0.090t = 0, 0.010, \ldots, 0.090t = 0, 0.010, \ldots, 0.090
) on a single figure, using the Viridis colormap to differentiate time steps. The x-axis is emergent space (x), the y-axis is probability density, and a legend labels each time step.

What the Simulation Shows
This simulation demonstrates the evolution of interference patterns in emergent spacetime ((x)) arising from two wavepackets in internal space (ξ\xi\xi
), akin to a double-slit experiment. Here’s what it shows, based on the code and expected output:
Initial State in Internal Space:
The initial plot of ∣ψ(ξ,0)∣2|\psi(\xi, 0)|^2|\psi(\xi, 0)|^2
 shows two Gaussian peaks at ξ=−4.0\xi = -4.0\xi = -4.0
 and ξ=4.0\xi = 4.0\xi = 4.0
, each with width σ=0.5\sigma = 0.5\sigma = 0.5
. The peaks are symmetric, with a separation of 8 units, mimicking two slits in a double-slit setup.

The phase factor eik0ξe^{i k_0 \xi}e^{i k_0 \xi}
 (with k0=5.0k_0 = 5.0k_0 = 5.0
) gives each wavepacket a momentum, setting up interference when they overlap in emergent space.

Free Evolution in Internal Space:
Since κ=0\kappa = 0\kappa = 0
, the wavefunction evolves freely under the kinetic term:
iℏ∂ψ∂t=−ℏ22mξ∂2ψ∂ξ2i \hbar \frac{\partial \psi}{\partial t} = -\frac{\hbar^2}{2 m_\xi} \frac{\partial^2 \psi}{\partial \xi^2}i \hbar \frac{\partial \psi}{\partial t} = -\frac{\hbar^2}{2 m_\xi} \frac{\partial^2 \psi}{\partial \xi^2}

The two wavepackets spread and overlap over time due to dispersion. By t=0.1t = 0.1t = 0.1
, the Gaussians (initially σ=0.5\sigma = 0.5\sigma = 0.5
) widen significantly, as the spread of a Gaussian wavepacket in free evolution scales as:
σ(t)=σ02+(ℏtmξσ0)2\sigma(t) = \sqrt{\sigma_0^2 + \left(\frac{\hbar t}{m_\xi \sigma_0}\right)^2}\sigma(t) = \sqrt{\sigma_0^2 + \left(\frac{\hbar t}{m_\xi \sigma_0}\right)^2}
With σ0=0.5\sigma_0 = 0.5\sigma_0 = 0.5
, ℏ=1\hbar = 1\hbar = 1
, mξ=1m_\xi = 1m_\xi = 1
, t=0.1t = 0.1t = 0.1
:
σ(0.1)=0.52+(0.10.5)2=0.25+0.04≈0.538\sigma(0.1) = \sqrt{0.5^2 + \left(\frac{0.1}{0.5}\right)^2} = \sqrt{0.25 + 0.04} \approx 0.538\sigma(0.1) = \sqrt{0.5^2 + \left(\frac{0.1}{0.5}\right)^2} = \sqrt{0.25 + 0.04} \approx 0.538
The spread is small over 0.1 time units, but the wavepackets begin to overlap, especially in the center (ξ=0\xi = 0\xi = 0
).

Interference in Emergent Space:
The Fourier transform maps ψ(ξ,t)\psi(\xi, t)\psi(\xi, t)
 to Ψ(x,t)\Psi(x, t)\Psi(x, t)
, producing an interference pattern in emergent space. The mapping kx=βk0−xk_x = \beta k_0 - xk_x = \beta k_0 - x
 (with β=1\beta = 1\beta = 1
, k0=5.0k_0 = 5.0k_0 = 5.0
) centers the pattern around x=0x = 0x = 0
 (since kx=0k_x = 0k_x = 0
 when x=βk0=5.0x = \beta k_0 = 5.0x = \beta k_0 = 5.0
, but the transform adjusts the phase across (x)).

Expected Pattern: The output plot of ∣Ψ(x,t)∣2|\Psi(x, t)|^2|\Psi(x, t)|^2
 shows a double-slit interference pattern—an envelope (from the Gaussian Fourier transform) modulated by interference fringes (from the phase difference between the two wavepackets). The fringe spacing is determined by the separation of the wavepackets (Δξ=8\Delta \xi = 8\Delta \xi = 8
) and the wavenumber:
Fringe spacing≈2πΔξ⋅β=2π8⋅1=π4≈0.785\text{Fringe spacing} \approx \frac{2\pi}{\Delta \xi} \cdot \beta = \frac{2\pi}{8} \cdot 1 = \frac{\pi}{4} \approx 0.785\text{Fringe spacing} \approx \frac{2\pi}{\Delta \xi} \cdot \beta = \frac{2\pi}{8} \cdot 1 = \frac{\pi}{4} \approx 0.785
However, the mapping kx=βk0−xk_x = \beta k_0 - xk_x = \beta k_0 - x
 adjusts this based on the emergent space scaling, so we’d expect a slightly different spacing (around 1.257, based on λ=2π/k0=2π/5≈1.257\lambda = 2\pi/k_0 = 2\pi/5 \approx 1.257\lambda = 2\pi/k_0 = 2\pi/5 \approx 1.257
).

Evolution Over Time: As the wavepackets spread in internal space, the interference pattern in emergent space evolves:
At t=0t = 0t = 0
, the pattern is sharp, with clear fringes due to the distinct wavepackets.

By t=0.090t = 0.090t = 0.090
, the spreading wavepackets overlap more, potentially washing out the fringes as the envelope broadens. The simulation doesn’t erase which-path information (unlike your previous quantum eraser simulation), so we expect the interference to persist but evolve in amplitude and width.

Visual Output:
The emergent space plot shows ∣Ψ(x,t)∣2|\Psi(x, t)|^2|\Psi(x, t)|^2
 for t=0,0.010,…,0.090t = 0, 0.010, \ldots, 0.090t = 0, 0.010, \ldots, 0.090
, with each time step in a different color (Viridis colormap). The interference pattern starts with distinct fringes and evolves as the wavepackets spread, showing how the emergent universe’s probability density changes over time.

Expected Features:
Fringes: You’d see ~16 fringes across x∈[−10,10]x \in [-10, 10]x \in [-10, 10]
 (20 units, with spacing ~1.257, so 20/1.257≈1620 / 1.257 \approx 1620 / 1.257 \approx 16
).

Envelope: The Gaussian envelope (from the Fourier transform of the Gaussians) constrains the fringes, with a width related to σ=0.5\sigma = 0.5\sigma = 0.5
.

Evolution: The pattern broadens over time, with fringes potentially becoming less distinct as the wavepackets overlap more in internal space.

'''


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Parameters
M = 4096  # Grid points in internal space
L = 20.0  # Internal space range [-L, L]
dx = 2 * L / M  # Internal grid spacing
J = 4096  # Emergent space grid points
X = 20.0  # Emergent space range [-X, X]
dx_emergent = 2 * X / J  # Emergent grid spacing
dt = 0.001  # Time step
beta = 1.0  # Mapping scale (adjusted to center and spread fringes)
hbar = 1.0  # Planck constant
m_xi = 1.0  # Effective mass
kappa = 0.0  # Potential strength (set to 0 for free evolution)
num_steps = 100  # Number of time steps
plot_interval = 10  # Plot every 10 steps

# Internal space grid
xi = np.linspace(-L, L, M, endpoint=False)
# Emergent space grid
x = np.linspace(-X, X, J, endpoint=False)

# Initialize state function: two Gaussian wavepackets (double-slit-like)
sigma = 0.5  # Width of wavepackets
xi1, xi2 = -4.0, 4.0  # Centers of wavepackets
k0 = 5.0  # Central wavenumber (adjusted to center pattern)
A = 1.0 / np.sqrt(2 * np.pi * sigma**2)  # Normalization factor
psi = A * (np.exp(-(xi - xi1)**2 / (2 * sigma**2)) * np.exp(1j * k0 * xi) +
           np.exp(-(xi - xi2)**2 / (2 * sigma**2)) * np.exp(1j * k0 * xi))
psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)  # Normalize

# Plot initial internal state to verify setup
plt.figure()
plt.plot(xi, np.abs(psi)**2, label="Initial |ψ(ξ,0)|²")
plt.xlabel("Internal Space (ξ)")
plt.ylabel("Probability Density")
plt.title("Initial State in Internal Space")
plt.legend()
plt.grid(True)
plt.show()

# Potential: set to 0 for free evolution
V = kappa * xi**2

# Precompute kinetic term in momentum space
kx = 2 * np.pi * np.fft.fftfreq(M, dx)
kinetic = hbar**2 * kx**2 / (2 * m_xi)

# Set up the colormap for different time steps
colors = cm.viridis(np.linspace(0, 1, num_steps // plot_interval))

# Create a single figure for all time steps
plt.figure(figsize=(10, 6))
plt.xlabel("Emergent Space (x)")
plt.ylabel("Probability Density |Ψ(x,t)|²")
plt.title("Emergent Universe: Interference Pattern Evolution")
plt.grid(True)

# Time evolution using split-operator method
for step in range(num_steps):
    t = step * dt

    # Apply potential term
    psi = np.exp(-1j * dt * V / hbar) * psi

    # Apply kinetic term in momentum space
    psi_fft = np.fft.fft(psi)
    psi_fft = np.exp(-1j * dt * kinetic / hbar) * psi_fft
    psi = np.fft.ifft(psi_fft)

    # Renormalize to prevent numerical drift
    psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)

    # Plot every `plot_interval` steps
    if step % plot_interval == 0:
        # Fourier transform to emergent space
        Psi = np.zeros(J, dtype=complex)
        for j in range(J):
            # Compute Ψ(x, t) = ∫ ψ(ξ, t) e^(-i k_x ξ) dξ, where k_x = x / β
            kx =(beta * k0 - x[j])  # Adjusted to center pattern
            Psi[j] = np.sum(psi * np.exp(-1j * kx * xi)) * dx
        prob = np.abs(Psi)**2  # Probability density

        # Plot with a unique color for this time step
        color = colors[step // plot_interval]
        plt.plot(x, prob, label=f"Time: {t:.3f}", color=color)

# Add legend and show the plot
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.show()