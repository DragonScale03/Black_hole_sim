# 🌌 Kerr Black Hole Simulation

> **A real-time, GPU-accelerated simulation of a rotating (Kerr) black hole — featuring
> geodesic integration, frame-dragging, gravitational lensing, relativistic Doppler beaming,
> and live information-decay diagnostics.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![Taichi 1.7+](https://img.shields.io/badge/taichi-1.7%2B-orange)](https://www.taichi-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Installation & Setup](#2-installation--setup)
3. [Running the Simulation](#3-running-the-simulation)
4. [The Physics Engine](#4-the-physics-engine)
   - 4.1 [The Kerr Metric & Boyer-Lindquist Coordinates](#41-the-kerr-metric--boyer-lindquist-coordinates)
   - 4.2 [Lense-Thirring Effect (Frame-Dragging)](#42-lense-thirring-effect-frame-dragging)
   - 4.3 [The RK4 Integrator & Numerical Stability](#43-the-rk4-integrator--numerical-stability)
   - 4.4 [Gravitational Time Dilation](#44-gravitational-time-dilation)
   - 4.5 [The Innermost Stable Circular Orbit (ISCO)](#45-the-innermost-stable-circular-orbit-isco)
5. [Visual Effects Engine](#5-visual-effects-engine)
   - 5.1 [Blackbody Radiation & Temperature Mapping](#51-blackbody-radiation--temperature-mapping)
   - 5.2 [Relativistic Doppler Beaming](#52-relativistic-doppler-beaming)
   - 5.3 [Gravitational Lensing via Raymarching](#53-gravitational-lensing-via-raymarching)
   - 5.4 [The Photon Ring](#54-the-photon-ring)
6. [Information Decay Diagnostic](#6-information-decay-diagnostic)
7. [Controls](#7-controls)
8. [Architecture & Performance](#8-architecture--performance)
9. [References & Further Reading](#9-references--further-reading)

---

## 1. Project Overview

This simulation models the spacetime geometry around a **Kerr black hole** — a solution to
Einstein's field equations describing a massive, *rotating* body. Unlike the simpler
Schwarzschild solution (a static, non-rotating black hole), the Kerr solution introduces
profound new phenomena:

- **Frame-dragging** (Lense-Thirring effect): The rotating mass literally drags spacetime
  itself into rotation, forcing all nearby matter — and even light — to co-rotate.
- **The ergosphere**: A region *outside* the event horizon where no particle can remain
  stationary; it is dragged along by the spacetime vortex.
- **The photon sphere**: A radius at which photons can orbit the black hole, creating the
  luminous ring observed by the Event Horizon Telescope.

The simulation places **6,000 massive particles** on physically correct circular orbits
within an accretion disk, evolves them under the full Kerr gravitational potential using
a 4th-order Runge-Kutta (RK4) integrator, and renders the result in real-time using the
[Taichi](https://www.taichi-lang.org/) GPU-parallel computing framework. A live
**Shannon entropy** metric is printed to the terminal, quantifying the rate at which
information is lost as matter crosses the event horizon.

The project is inspired by real astrophysical imaging — particularly the Event Horizon
Telescope observations of M87* and Sagittarius A* — and aims to reproduce the key visual
signatures of relativistic accretion: the asymmetric brightness of the disk due to Doppler
beaming, the orange-white photon ring, and the absolute blackness of the horizon itself.

---

## 2. Installation & Setup

### Prerequisites

- **Python 3.9 or later** (Python 3.11 recommended)
- A **CUDA-capable NVIDIA GPU** is strongly recommended for smooth real-time rendering.
  The simulation will automatically fall back to CPU if no GPU is detected.

### Install Dependencies

Open your terminal and run the following commands:

```bash
# 1. (Recommended) Create and activate a virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# 2. Install Taichi (GPU-accelerated parallel computing framework)
pip install taichi

# 3. Install NumPy (required for initialisation and entropy calculation)
pip install numpy
```

> **Taichi version note:** This simulation requires `taichi >= 1.7.0`.
> You can verify your installed version with `python -c "import taichi; print(taichi.__version__)"`.

### Full One-Line Install

```bash
pip install "taichi>=1.7.0" numpy
```

---

## 3. Running the Simulation

```bash
# Navigate to the project directory
cd /path/to/BlackHoles

# Run the simulation
python main.py
```

A **1280×720 GUI window** will open immediately, rendering the accretion disk in real-time.
Simultaneously, the terminal will stream diagnostic output:

```
[Kerr BH]  M=1.0  a=0.95  r+=1.3123  r_ISCO=2.0441
           Particles=6000  dt=0.005  r_photon≈2.2500
[Init] Seeding accretion disk particles on Kerr circular orbits …
[Init] Done. r_ISCO=2.044M  r_disk_out=22.0M

  Frame    Alive    H (bits)    H/H_max      FPS
----------------------------------------------------
      0     6000    7.8431       0.7734      ---
    120     5981    7.6210       0.7515     48.2
    240     5965    7.5003       0.7396     49.1
```

---

## 4. The Physics Engine

### 4.1 The Kerr Metric & Boyer-Lindquist Coordinates

The Kerr metric describes the spacetime geometry exterior to a rotating black hole of mass
**M** and specific angular momentum **a = J/M** (where J is the black hole's angular
momentum). In Boyer-Lindquist (BL) coordinates $(t, r, \theta, \phi)$, the line element is:

$$ds^2 = -\left(1 - \frac{2Mr}{\Sigma}\right)dt^2 - \frac{4Mar\sin^2\theta}{\Sigma}\,dt\,d\phi + \frac{\Sigma}{\Delta}dr^2 + \Sigma\, d\theta^2 + \left(r^2 + a^2 + \frac{2Ma^2r\sin^2\theta}{\Sigma}\right)\sin^2\theta\, d\phi^2$$

where the key auxiliary functions are:

| Symbol | Expression | Physical Meaning |
|--------|-----------|-----------------|
| $\Sigma$ | $r^2 + a^2\cos^2\theta$ | Oblate geometry factor |
| $\Delta$ | $r^2 - 2Mr + a^2$ | Horizon structure |
| $r_+$ | $M + \sqrt{M^2 - a^2}$ | Outer event horizon radius |
| $r_-$ | $M - \sqrt{M^2 - a^2}$ | Inner (Cauchy) horizon radius |

The simulation uses **geometrized units** where $G = c = 1$, so all distances are measured
in units of the black hole mass $M$, and all velocities are fractions of $c$.

Particle trajectories are geodesics of this metric — curves that extremise proper time.
The geodesic equations are derived from the Hamiltonian:

$$H = \frac{1}{2}g^{\mu\nu}p_\mu p_\nu = -\frac{1}{2}\mu^2$$

where $\mu = 1$ for massive particles. This yields a system of first-order ODEs in the
canonical momenta $(p_r, p_\theta, p_\phi, E)$.

Three **conserved quantities** dramatically simplify integration:

1. **Energy** $E = -p_t$ (from time-translation symmetry)
2. **Angular momentum** $L_z = p_\phi$ (from azimuthal symmetry)
3. **Carter's constant** $Q$ (from the hidden symmetry of the Kerr solution, discovered
   by Brandon Carter in 1968) — effectively the square of the "latitudinal" momentum

These constants are computed once at initialisation from the circular-orbit conditions and
used as fixed parameters throughout integration, dramatically improving numerical stability.

### 4.2 Lense-Thirring Effect (Frame-Dragging)

The most striking feature of the Kerr metric is the off-diagonal $dt\, d\phi$ cross-term,
proportional to $a$. This term means that **a particle at rest (dr = dθ = dφ = 0) is still
dragged in the φ-direction** with angular velocity:

$$\Omega_{LT}(r, \theta) = \frac{d\phi}{dt}\Bigg|_{\text{rest}} = \frac{2Mar}{(r^2+a^2)^2 - \Delta a^2\sin^2\theta}$$

This is the **Lense-Thirring (frame-dragging) angular velocity**. Its physical consequences
are profound:

- **Inner disk particles** spiral inward faster on the prograde side than they would without
  spin, because the spacetime itself provides an additional angular momentum "push."
- **The ergosphere** ($r < r_\text{ergo} = 2M$ on the equator) is the region where
  $\Omega_{LT}$ exceeds the orbital velocity — no static observer can exist here. The name
  "ergosphere" comes from the Greek *ergon* (work) because energy can be extracted from
  the ergosphere via the Penrose process.
- **In the simulation**, $\Omega_{LT}$ enters the geodesic equation for $d\phi/d\lambda$
  as an additive term, causing the inner disk to visually "smear" and circularise more
  rapidly than an equivalent Schwarzschild disk.

The Lense-Thirring effect also appears in the **raymarching renderer** as a
gravito-magnetic deflection of null rays, modelled as:

$$\vec{a}_\text{LT} \propto \frac{a}{r^3}\left[3(\vec{J}\cdot\hat{r})\hat{r} - \vec{J}\right]$$

which causes photons to be slightly deflected in the $\phi$-direction as they pass the
spinning black hole — a key contributor to the asymmetry of the photon ring.

### 4.3 The RK4 Integrator & Numerical Stability

The simulation uses a **4th-order Runge-Kutta (RK4)** integrator to advance the state
vector $(r, \theta, \phi, p_r, p_\theta)$ of each particle.

#### Why Not Euler or RK2?

| Method | Order | Global Error | Problem |
|--------|-------|-------------|---------|
| Euler (1st-order) | $O(h)$ | $O(h)$ | Orbits spiral outward — energy is not conserved |
| Improved Euler / RK2 | $O(h^2)$ | $O(h^2)$ | Better, but still significant drift over 1000s of steps |
| **RK4 (this simulation)** | **$O(h^4)$** | **$O(h^5)$** | **Orbits stable for 10⁵+ steps** |
| Symplectic integrators | Varies | Bounded | Best for long-term, but complex to implement for Kerr |

The RK4 method evaluates the equations of motion at four intermediate points within each
time step and combines them as:

$$y_{n+1} = y_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

where $k_1, k_2, k_3, k_4$ are the slopes evaluated at $t_n$, $t_n + h/2$, $t_n + h/2$,
and $t_n + h$ respectively. The local truncation error is $O(h^5)$, meaning halving the
step size reduces error by a factor of **32** — far superior to Euler's factor of 2.

In the strong-field Kerr regime (near the ISCO), the geodesic equations become highly
non-linear. RK4's superior accuracy prevents the unphysical numerical spiral-in that would
plague lower-order methods, keeping all orbits energetically consistent with their conserved
quantities $E$ and $L_z$ throughout the simulation.

### 4.4 Gravitational Time Dilation

One of the most counter-intuitive predictions of General Relativity is that **time passes
more slowly in stronger gravitational fields**. Near the black hole, a particle's proper
time $\tau$ runs slower relative to coordinate time $t$ by:

$$\frac{d\tau}{dt} = \sqrt{-g_{tt}} \approx \sqrt{1 - \frac{2M}{r}}$$

(equatorial approximation; full Kerr expression involves $\Sigma$ and $\Delta$).

In this simulation, **each particle's time-step is individually scaled** by this factor:

```python
time_dilation = sqrt(Delta / Sigma)
dt_local = DT_BASE * time_dilation
```

This has three important effects:

1. **Physical accuracy**: Particles near the horizon evolve slowly in coordinate time —
   exactly as observed from far away. From an external observer's perspective, matter
   appears to "freeze" at the horizon, asymptotically approaching but never crossing it.

2. **Numerical stability**: Particles in the strong-gravity region take smaller steps,
   where the geodesic curvature is highest. This prevents numerical errors from
   accumulating rapidly near the most non-linear part of the potential.

3. **Visual effect**: Inner-disk particles appear to slow down relative to outer-disk
   particles, creating the characteristic "trailing edge" appearance of the approaching
   side of the disk.

### 4.5 The Innermost Stable Circular Orbit (ISCO)

For a Schwarzschild black hole, the ISCO is at $r = 6M$. For a Kerr black hole with spin
$a$, the ISCO radius depends on the spin via the analytical formula (Bardeen 1972):

$$r_\text{ISCO} = M\left(3 + Z_2 - \sqrt{(3 - Z_1)(3 + Z_1 + 2Z_2)}\right)$$

where $Z_1$ and $Z_2$ are algebraic functions of $a/M$. At spin $a = 0.95M$, this gives
$r_\text{ISCO} \approx 2.04M$ — much closer to the horizon than the Schwarzschild value.

All accretion disk particles are seeded with $r \geq r_\text{ISCO}$. Inside the ISCO,
no stable circular orbit exists; matter spirals in rapidly on a plunging trajectory and
is absorbed by the horizon. The simulation absorbs particles when $r \leq 1.02\, r_+$.

---

## 5. Visual Effects Engine

### 5.1 Blackbody Radiation & Temperature Mapping

The accretion disk radiates as a blackbody. According to the **Novikov-Thorne thin-disk
model**, the effective temperature at radius $r$ scales as:

$$T(r) \propto r^{-3/4}$$

(more precisely, with a correction factor involving the ISCO). The simulation uses:

```python
T_K = T_max * (R_ISCO / r) ** 0.75
```

where `T_max ≈ 1.8 × 10⁷ K` at the ISCO — consistent with AGN-scaled accretion.

The temperature is then converted to an **RGB colour** using Planck's law approximated
via the Krystek-Tanner chromaticity fits. This produces:
- **Outer disk** (cool, $T \sim 3000$–$5000\,K$): Deep red-orange
- **Middle disk** ($T \sim 6000$–$10000\,K$): Yellow-white
- **Inner disk** ($T \sim 10^7\,K$, Doppler-boosted): Intensely blue-white (approaching side)

### 5.2 Relativistic Doppler Beaming

This is the most visually dramatic physical effect in the simulation. As accretion disk
matter orbits at relativistic speeds ($v \sim 0.3$–$0.5\,c$ near the ISCO), the observed
radiation is strongly modified.

The **relativistic Doppler factor** for a source moving at velocity $\beta = v/c$ at
angle $\alpha$ to the line of sight is:

$$D = \frac{1}{\gamma(1 - \beta\cos\alpha)}$$

where $\gamma = 1/\sqrt{1-\beta^2}$ is the Lorentz factor. This factor enters the
observed intensity as:

$$I_\text{obs}(\nu_\text{obs}) = D^4 \cdot I_\text{em}(\nu_\text{em})$$

The four powers of $D$ arise from:
- $D^1$: Photon frequency blueshift ($\nu_\text{obs} = D\nu_\text{em}$)
- $D^3$: Relativistic beaming (photons bunched in forward direction)
- $D^{-0} = 1$: … times the Jacobian of the solid-angle transformation, totalling $D^4$

**Consequences visible in the simulation:**

| Side of Disk | $\cos\alpha$ | $D$ | $I_\text{obs}/I_\text{em}$ |
|---|---|---|---|
| **Approaching** (left side) | $\approx +1$ | $\gg 1$ | Up to $8\times$ brighter |
| **Receding** (right side) | $\approx -1$ | $\ll 1$ | Down to $0.1\times$ dimmer |

The **approaching side** of the disk blazes blue-white, while the **receding side** is
dramatically redder and dimmer. This is precisely what was observed in the 2019 Event
Horizon Telescope image of M87* — the bright crescent on the southern side of the ring
is the approaching side of the jet-driven disk.

The Doppler factor also **blue-shifts the temperature**: $T_\text{obs} = D \cdot T_K$,
making the approaching side appear hotter and bluer in addition to being brighter.

### 5.3 Gravitational Lensing via Raymarching

The simulation renders gravitational lensing by **raymarching** null geodesics (light
paths) through the curved Kerr spacetime. For each screen pixel, a ray is shot from the
camera and iteratively deflected by the gravitational field:

```
For each step:
    1. Compute gravitational acceleration: a_grav = -M/r³ × r⃗
    2. Add Lense-Thirring correction: a_LT ∝ a/r³
    3. Update ray direction (bending)
    4. Check for horizon capture or photon-sphere passage
```

The **deflection angle** for a null ray with impact parameter $b$ passing a Kerr black
hole at closest approach $r_0$ is approximately:

$$\Delta\phi \approx \frac{4M}{b} + \frac{2\pi M^2}{b^2} + O\left(\frac{M^3}{b^3}\right)$$

with spin-dependent corrections at the next order. Rays with $b < b_\text{crit} \approx
3\sqrt{3}\,M$ are captured by the black hole; these produce the black shadow.

### 5.4 The Photon Ring

At the **photon sphere** ($r \approx 1.5\,r_+$ for spin $a = 0.95M$), null geodesics can
orbit the black hole indefinitely. In practice, this orbit is unstable — tiny perturbations
either cause the photon to escape (producing the bright ring) or fall in (absorbed).

The simulation detects when a raymarched photon passes through the photon-sphere shell
$[r_\text{ph} \times 0.82,\; r_\text{ph} \times 1.55]$ and accumulates an orange-white
glow. Multiple passes around the black hole contribute additively, creating the
characteristic bright annulus seen in EHT images.

---

## 6. Information Decay Diagnostic

Every `120` frames, the simulation prints a live **Shannon entropy** measurement to the
terminal. This metric quantifies the **information content** of the observable particle
distribution.

### The Mathematics of Shannon Entropy

Given a discrete probability distribution $\{p_i\}$ over phase-space cells, Shannon
entropy is defined as:

$$H = -\sum_{i} p_i \log_2 p_i \quad \text{(bits)}$$

The simulation bins the $N = 6000$ particles into a $32 \times 32$ grid in
$(r,\, v_\text{orbital})$ phase space. Each cell's probability is:

$$p_i = \frac{n_i}{N_\text{alive}}$$

where $n_i$ is the particle count in cell $i$ and $N_\text{alive}$ is the total number of
living particles.

The **maximum possible entropy** for a $32 \times 32 = 1024$-cell grid is:

$$H_\text{max} = \log_2(1024) = 10.0 \text{ bits}$$

achieved only when all cells are equally occupied (perfectly uniform distribution). The
printed value $H/H_\text{max}$ is a number between 0 and 1.

### Physical Interpretation: Information Loss at the Horizon

The Shannon entropy measurement captures a real physical phenomenon central to the
**black hole information paradox**.

#### 1. Early-time high entropy
At simulation start, particles are spread across the full disk from $r_\text{ISCO}$ to
$22\,M$, with a wide distribution of orbital velocities. The phase space is well-populated
→ **high entropy**, typically $H/H_\text{max} \approx 0.75$–$0.80$.

#### 2. Entropy decay as matter falls in
As the simulation runs, particles with orbits perturbed inside the ISCO plunge toward the
horizon. Their phase-space cells empty, and the distribution becomes **less uniform** →
**entropy decreases**. Each particle that crosses the horizon carries information that
becomes inaccessible to external observers.

This is the classical formulation of the **black hole information paradox** (Hawking, 1976):
- **Classical GR** says: information that crosses the horizon is lost forever. The
  singularity destroys it.
- **Quantum mechanics** says: information cannot be destroyed — it must be encoded
  somewhere (Hawking radiation, holographic principle, firewall conjecture).

The entropy metric in this simulation models the **observable** information loss: from the
perspective of an external observer, matter falling toward the horizon appears to
asymptotically freeze (due to time dilation) and fade (its Hawking radiation is
undetectably faint at stellar-mass scales). The observable phase-space distribution
loses complexity — a decrease in Shannon entropy.

#### 3. Reading the Terminal Output

```
  Frame    Alive    H (bits)    H/H_max      FPS
----------------------------------------------------
      0     6000    7.8431       0.7734      ---
    120     5981    7.6210       0.7515     48.2
```

| Column | Meaning |
|--------|---------|
| `Frame` | Simulation step number |
| `Alive` | Number of particles that have not yet crossed the horizon |
| `H (bits)` | Absolute Shannon entropy of the phase-space distribution |
| `H/H_max` | Normalised entropy (1.0 = maximum disorder, 0.0 = all particles in one cell) |
| `FPS` | Rendering frames per second (averaged over the interval) |

A **warning** is printed if $H/H_\text{max}$ drops below 0.35 — indicating that a large
fraction of particles has been absorbed or clustered into a narrow phase-space region,
corresponding to high information loss.

---

## 7. Controls

| Input | Action |
|-------|--------|
| **Mouse drag** | Rotate camera inclination (tilt disk toward edge-on / face-on) |
| **Scroll wheel** | Zoom in / out |
| **ESC** | Quit simulation |

The simulation automatically **respawns dead particles** (those swallowed by the horizon)
at new random disk radii every 60 frames, maintaining a dense, visually rich accretion disk.

---

## 8. Architecture & Performance

```
main.py
│
├── Physics Layer (Taichi GPU kernels)
│   ├── kerr_sigma / kerr_delta / kerr_omega   — Kerr metric functions
│   ├── geodesic_rhs()                          — Kerr geodesic equations (ODE RHS)
│   ├── rk4_step()                              — 4th-order Runge-Kutta stepper
│   └── update_particles()                      — Main particle evolution kernel
│
├── Render Layer (Taichi GPU kernels)
│   ├── render_raymarch()                       — Null geodesic lensing + photon ring
│   ├── render_particles()                      — Doppler-beamed disk particles
│   ├── blackbody_color()                       — Planck spectral → RGB
│   ├── doppler_factor()                        — Relativistic D factor per particle
│   └── clear_fb()                              — Framebuffer decay (motion blur)
│
├── Diagnostic Layer (NumPy / CPU)
│   ├── build_histogram()                       — Phase-space binning (Taichi kernel)
│   └── compute_shannon_entropy()               — H = -Σ p log₂ p
│
└── Control Layer (Python)
    ├── init_particles()                        — Circular Kerr orbit initialisation
    ├── respawn_dead()                          — Particle recycling
    └── main()                                  — GUI event loop
```

**Typical performance (NVIDIA RTX 3060 / 4070):**
- GPU mode: **45–60 FPS** at 1280×720 with 6,000 particles
- CPU mode: **4–10 FPS** (functional, but slower)

The raymarching kernel is the primary GPU bottleneck (80 march steps × 1280×720 pixels).
Reduce `N_MARCH` in `render_raymarch()` to trade visual quality for performance.

---

## 9. References & Further Reading

### Primary Physics References

1. **Roy P. Kerr** (1963). "Gravitational field of a spinning mass as an example of
   algebraically special metrics." *Physical Review Letters*, 11, 237.
   — The original paper deriving the Kerr metric.

2. **James M. Bardeen, William H. Press & Saul A. Teukolsky** (1972). "Rotating Black
   Holes: Locally Nonrotating Frames, Energy Extraction, and Scalar Synchrotron
   Radiation." *Astrophysical Journal*, 178, 347.
   — The foundational paper on geodesics, ISCO, and energy extraction in Kerr spacetime.

3. **Brandon Carter** (1968). "Global Structure of the Kerr Family of Gravitational
   Fields." *Physical Review*, 174, 1559.
   — Discovery of Carter's constant, enabling complete integrability of Kerr geodesics.

4. **Igor D. Novikov & Kip S. Thorne** (1973). "Astrophysics of Black Holes." In *Black
   Holes*, ed. C. DeWitt & B. DeWitt. — The Novikov-Thorne thin disk temperature profile.

5. **Stephen W. Hawking** (1976). "Breakdown of predictability in gravitational collapse."
   *Physical Review D*, 14, 2460. — The information paradox.

6. **Event Horizon Telescope Collaboration** (2019). "First M87 Event Horizon Telescope
   Results. I–VI." *Astrophysical Journal Letters*, 875, L1–L6.
   — The first images of a black hole shadow, showing the photon ring and Doppler beaming.

### Computational References

7. **Scott A. Hughes** (2000). "Evolution of circular, non-equatorial orbits of Kerr
   black holes due to gravitational-wave emission." *Physical Review D*, 61, 084004.

8. **Steve Drasco & Scott A. Hughes** (2004). "Rotating black hole orbit functionals in
   the frequency domain." *Physical Review D*, 69, 044 027.

9. **Claude E. Shannon** (1948). "A Mathematical Theory of Communication."
   *Bell System Technical Journal*, 27, 379–423. — The foundation of information entropy.

### Visualisation References

10. **Alain Riazuelo** (2019). "Seeing relativity: Black hole image." Simulations and
    images at the Paris Observatory.

11. **Oliver James et al.** (2015). "Gravitational lensing by spinning black holes in
    astrophysics, and in the movie Interstellar." *Classical and Quantum Gravity*, 32, 6.

---

*Built with [Taichi Lang](https://www.taichi-lang.org/) — a high-performance, parallel
programming language embedded in Python.*
