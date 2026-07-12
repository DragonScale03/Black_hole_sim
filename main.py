"""
Schwarzschild Black Hole — True 3D Particle Geodesic Simulation
=========================================================
Schwarzschild metric (Eq. 1, Nappo & Cangiotti 2024):
  ds² = -(1-2M/R)dt² + (1-2M/R)^{-1}dr² + r²(dθ² + sin²θdφ²)

Every particle follows the EXACT equatorial geodesic.
3D Camera added to allow flying around the black hole.

Dependencies: taichi >= 1.7.0, numpy
"""

import taichi as ti
import numpy as np
import math, time, sys

# ─── Init ─────────────────────────────────────────────────────────────────────
try:
    ti.init(arch=ti.gpu, default_fp=ti.f32)
    BACKEND = "GPU"
except Exception:
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    BACKEND = "CPU"

# ─── Physical Constants (G=c=M=1) ─────────────────────────────────────────────
M          = 1.0
R_SCHWARZ  = 2.0 * M        # Schwarzschild radius / event horizon
R_PHOTON   = 3.0 * M        # photon sphere
R_ISCO     = 6.0 * M        # innermost stable circular orbit
R_OUT      = 40.0 * M       # outer disk edge

N_PART     = 2500
DT         = 0.25

WIN_W, WIN_H = 1280, 720

# ─── 3D Camera ────────────────────────────────────────────────────────────────
cam_radius = 42.0
cam_theta  = math.radians(60.0)
cam_phi    = 0.0

cam_pos_f = ti.Vector.field(3, ti.f32, shape=())
cam_s_f   = ti.Vector.field(3, ti.f32, shape=())
cam_u_f   = ti.Vector.field(3, ti.f32, shape=())
cam_f_f   = ti.Vector.field(3, ti.f32, shape=())
PPM_f     = ti.field(ti.f32, shape=())

def update_camera():
    global cam_radius, cam_theta, cam_phi
    cam_theta = max(min(cam_theta, math.pi - 0.01), 0.01)
    cam_radius = max(cam_radius, 4.0)

    cx = cam_radius * math.sin(cam_theta) * math.cos(cam_phi)
    cy = cam_radius * math.cos(cam_theta)
    cz = cam_radius * math.sin(cam_theta) * math.sin(cam_phi)
    
    pos = np.array([cx, cy, cz])
    target = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 1.0, 0.0])
    
    f = target - pos
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    
    cam_pos_f[None] = pos.astype(np.float32)
    cam_s_f[None] = s.astype(np.float32)
    cam_u_f[None] = u.astype(np.float32)
    cam_f_f[None] = f.astype(np.float32)
    
    fov_factor = 1.2
    PPM_f[None] = (1.0 / cam_radius) * fov_factor * (WIN_H * 0.5)

# ─── Rendering ────────────────────────────────────────────────────────────────
DECAY          = 0.80
I_BASE         = 0.20
N_STARS        = 400
DEBUG_COLOR    = "--debug-color" in sys.argv
ENTROPY_INTERVAL = 120

r_pos   = ti.field(ti.f32, N_PART)
ph_pos  = ti.field(ti.f32, N_PART)
r_dot   = ti.field(ti.f32, N_PART)
ang_mom = ti.field(ti.f32, N_PART)
alive   = ti.field(ti.i32, N_PART)

star_vec = ti.Vector.field(3, ti.f32, N_STARS)
star_br  = ti.field(ti.f32, N_STARS)

fb       = ti.Vector.field(3, ti.f32, (WIN_W, WIN_H))
N_BINS   = 32
hist_buf = ti.field(ti.i32, (N_BINS, N_BINS))

# ─── Geodesics ────────────────────────────────────────────────────────────────
@ti.func
def geo_rhs(r, rd, L):
    r2  = r*r; r3 = r2*r; r4 = r3*r; L2 = L*L
    return rd, L/r2, -M/r2 + L2/r3 - 3.0*M*L2/r4

@ti.func
def rk4(i, h):
    r = r_pos[i]; ph = ph_pos[i]; rd = r_dot[i]; L = ang_mom[i]
    k1r,k1p,k1d = geo_rhs(r,           rd,           L)
    k2r,k2p,k2d = geo_rhs(r+.5*h*k1r, rd+.5*h*k1d, L)
    k3r,k3p,k3d = geo_rhs(r+.5*h*k2r, rd+.5*h*k2d, L)
    k4r,k4p,k4d = geo_rhs(r+   h*k3r, rd+   h*k3d, L)
    r_pos[i]  = r  + h/6.0*(k1r+2.0*k2r+2.0*k3r+k4r)
    ph_pos[i] = ph + h/6.0*(k1p+2.0*k2p+2.0*k3p+k4p)
    r_dot[i]  = rd + h/6.0*(k1d+2.0*k2d+2.0*k3d+k4d)

@ti.kernel
def update_particles():
    PI2 = 2.0 * 3.14159265
    for i in range(N_PART):
        if alive[i] == 0:
            continue
        rk4(i, DT)
        ph = ph_pos[i]
        ph_pos[i] = ph - PI2 * ti.floor(ph / PI2)
        r = r_pos[i]
        if r <= R_SCHWARZ * 1.05 or r > R_OUT * 1.6:
            alive[i] = 0

@ti.kernel
def decay_fb():
    for px, py in fb:
        fb[px, py] = fb[px, py] * DECAY

@ti.func
def yw_color(T_norm):
    t   = ti.min(ti.max(T_norm, 0.0), 1.0)
    r_c = 1.0
    g_c = 0.55 + 0.45 * t
    b_c = 0.0  + 1.0  * (t*t*t*t)
    return ti.Vector([r_c, g_c, b_c])

@ti.func
def doppler_factor(r, ph):
    Omega = ti.sqrt(M / (r*r*r + 1e-10))
    beta  = ti.min(r * Omega, 0.94)
    vx = -ti.sin(ph)
    vy = 0.0
    vz = ti.cos(ph)
    px = r * ti.cos(ph)
    py = 0.0
    pz = r * ti.sin(ph)
    to_cam = (cam_pos_f[None] - ti.Vector([px, py, pz])).normalized()
    cos_alpha = vx*to_cam[0] + vy*to_cam[1] + vz*to_cam[2]
    gamma = 1.0 / ti.sqrt(ti.max(1.0 - beta*beta, 1e-5))
    return 1.0 / (gamma * (1.0 - beta * cos_alpha))

@ti.func
def project_3d(P_world):
    P_rel = P_world - cam_pos_f[None]
    x_c = P_rel.dot(cam_s_f[None])
    y_c = P_rel.dot(cam_u_f[None])
    z_c = P_rel.dot(cam_f_f[None])
    wx = -1.0; wy = -1.0
    if z_c > 0.1:
        fov_factor = 1.2
        aspect = float(WIN_W) / float(WIN_H)
        nx = (x_c / z_c) * fov_factor
        ny = (y_c / z_c) * fov_factor
        wx = nx / aspect * float(WIN_W) * 0.5
        wy = ny * float(WIN_H) * 0.5
    return wx, wy, z_c

# ─── Passes ───────────────────────────────────────────────────────────────────
@ti.kernel
def render_particles():
    cx = float(WIN_W) * 0.5
    cy = float(WIN_H) * 0.5
    ppM = PPM_f[None]
    R_E2 = (4.5 * M * ppM) * (4.5 * M * ppM)
    
    for i in range(N_PART):
        if alive[i] == 0: continue
        r = r_pos[i]; ph = ph_pos[i]
        T_norm = ti.min(ti.pow(R_ISCO / r, 0.75), 1.0)
        D = doppler_factor(r, ph)
        I = I_BASE * ti.min(D * D * D * D, 8.0)
        col = yw_color(T_norm)

        # 3D position in XZ plane
        P_world = ti.Vector([r * ti.cos(ph), 0.0, r * ti.sin(ph)])
        wx, wy, z_c = project_3d(P_world)
        if z_c < 0.1: continue

        b2 = wx*wx + wy*wy
        b  = ti.sqrt(b2) + 0.001
        disc   = ti.sqrt(b2 + 4.0*R_E2)
        b_plus = (b + disc) * 0.5
        b_min  = (disc - b) * 0.5
        
        ax_p = cx + wx / b * b_plus
        ay_p = cy + wy / b * b_plus
        px_p = int(ax_p); py_p = int(ay_p)
        if px_p >= 0 and px_p < WIN_W and py_p >= 0 and py_p < WIN_H:
            ti.atomic_add(fb[px_p, py_p], col * I)
            
        ax_m = cx - wx / b * b_min
        ay_m = cy - wy / b * b_min
        px_m = int(ax_m); py_m = int(ay_m)
        if px_m >= 0 and px_m < WIN_W and py_m >= 0 and py_m < WIN_H:
            ti.atomic_add(fb[px_m, py_m], col * I * 0.35)

@ti.kernel
def render_stars_lensed():
    cx = float(WIN_W) * 0.5
    cy = float(WIN_H) * 0.5
    ppM = PPM_f[None]
    sh  = 5.19615 * M * ppM
    R_E2 = (sh * 2.8) * (sh * 2.8)

    for s in range(N_STARS):
        # Distant star
        P_world = star_vec[s] * 1000.0
        wx, wy, z_c = project_3d(P_world)
        if z_c < 0.1: continue

        br = star_br[s]
        b2 = wx*wx + wy*wy
        b  = ti.sqrt(b2) + 0.001
        if b <= sh: continue

        disc = ti.sqrt(b2 + 4.0*R_E2)
        b_plus = (b + disc) * 0.5
        b_min  = (disc - b) * 0.5
        R_E = ti.sqrt(R_E2)
        u    = b / (R_E + 0.001)
        u2   = u * u
        mu_t = (u2 + 2.0) / (2.0 * u * ti.sqrt(u2 + 4.0) + 0.001)
        mu_p = ti.min(mu_t + 0.5, 25.0)
        mu_m = ti.max(mu_t - 0.5, 0.0)

        ax_p = cx + wx / b * b_plus
        ay_p = cy + wy / b * b_plus
        px_p = int(ax_p); py_p = int(ay_p)
        if px_p >= 0 and px_p < WIN_W and py_p >= 0 and py_p < WIN_H:
            sc = br * mu_p * 0.06
            ti.atomic_add(fb[px_p, py_p], ti.Vector([sc * 0.95, sc * 0.93, sc * 0.82]))

        if mu_m > 0.04:
            ax_m = cx - wx / b * b_min
            ay_m = cy - wy / b * b_min
            px_m = int(ax_m); py_m = int(ay_m)
            if px_m >= 0 and px_m < WIN_W and py_m >= 0 and py_m < WIN_H:
                sc2 = br * mu_m * 0.06
                ti.atomic_add(fb[px_m, py_m], ti.Vector([sc2*0.90, sc2*0.88, sc2*0.78]))

@ti.kernel
def render_horizon():
    cx = float(WIN_W) * 0.5; cy = float(WIN_H) * 0.5
    ppM = PPM_f[None]
    sh  = 5.19615 * M * ppM

    for px, py in fb:
        dx = float(px) - cx; dy = float(py) - cy
        dist = ti.sqrt(dx*dx + dy*dy)
        if dist <= sh:
            fb[px, py] = ti.Vector([0.0, 0.0, 0.0])
        elif dist < sh * 1.5:
            t = (dist - sh) / (sh * 0.5 + 0.001)
            s = ti.exp(-t * 8.0) * 0.08
            ti.atomic_add(fb[px, py], ti.Vector([s * 2.2, s * 1.75, s * 0.45]))

@ti.kernel
def tonemap():
    for px, py in fb:
        if ti.static(DEBUG_COLOR) and py < 30:
            t = float(px) / float(WIN_W)
            col = yw_color(t)
            fb[px, py] = ti.Vector([
                ti.pow(col[0], 1.0 / 2.2),
                ti.pow(col[1], 1.0 / 2.2),
                ti.pow(col[2], 1.0 / 2.2),
            ])
            continue
        c = fb[px, py]
        lum = 0.2126*c[0] + 0.7152*c[1] + 0.0722*c[2]
        lum_mapped = lum / (lum + 1.0)
        scale = lum_mapped / (lum + 1e-8)
        c = c * scale
        c_gamma = ti.Vector([
            ti.pow(ti.max(c[0], 0.0), 1.0 / 2.2),
            ti.pow(ti.max(c[1], 0.0), 1.0 / 2.2),
            ti.pow(ti.max(c[2], 0.0), 1.0 / 2.2),
        ])
        lum_gamma = 0.2126*c_gamma[0] + 0.7152*c_gamma[1] + 0.0722*c_gamma[2]
        boost = 1.35
        fb[px, py] = ti.max(lum_gamma + (c_gamma - lum_gamma) * boost, 0.0)

# ─── Init logic ───────────────────────────────────────────────────────────────
@ti.kernel
def build_histogram():
    for i, j in hist_buf:
        hist_buf[i, j] = 0
    for i in range(N_PART):
        if alive[i] == 0:
            continue
        ri = int((r_pos[i] - R_SCHWARZ) / (R_OUT - R_SCHWARZ + 1e-6) * N_BINS)
        lz = ti.abs(ang_mom[i]) / (r_pos[i] + 1e-6)
        vi = int(lz / 6.0 * N_BINS)
        ri = ti.min(ti.max(ri, 0), N_BINS - 1)
        vi = ti.min(ti.max(vi, 0), N_BINS - 1)
        ti.atomic_add(hist_buf[ri, vi], 1)

def compute_entropy():
    build_histogram()
    h = hist_buf.to_numpy().astype(np.float64)
    tot = h.sum()
    if tot < 1.0: return 0.0, 0.0
    p = h / tot
    p_nz = p[p > 0.0]
    H = -np.sum(p_nz * np.log2(p_nz))
    return H, H / math.log2(N_BINS * N_BINS)

def init_stars():
    rng = np.random.default_rng(99)
    # Generate points on a sphere
    s = np.zeros((N_STARS, 3), dtype=np.float32)
    b = np.clip(rng.exponential(0.55, N_STARS), 0.05, 2.0).astype(np.float32)
    for i in range(N_STARS):
        th = np.arccos(rng.uniform(-1, 1))
        ph = rng.uniform(0, 2*np.pi)
        s[i] = [math.sin(th)*math.cos(ph), math.cos(th), math.sin(th)*math.sin(ph)]
    star_vec.from_numpy(s)
    star_br.from_numpy(b)

def init_particles():
    print("[Init] Seeding particles ...")
    rng = np.random.default_rng(42)
    u = rng.uniform(0.0, 1.0, N_PART)
    r = R_ISCO * (R_OUT / R_ISCO)**u
    phi = rng.uniform(0.0, 2.0 * np.pi, N_PART)
    denom  = np.maximum(r - 3.0 * M, 0.10)
    L_circ = np.sqrt(M * r * r / denom)
    L_arr  = L_circ * (0.98 + rng.normal(0.0, 0.015, N_PART))
    rd_arr = -np.abs(rng.normal(0.0, 0.01, N_PART))
    r_pos.from_numpy(r.astype(np.float32))
    ph_pos.from_numpy(phi.astype(np.float32))
    r_dot.from_numpy(rd_arr.astype(np.float32))
    ang_mom.from_numpy(L_arr.astype(np.float32))
    alive.fill(1)

def respawn_dead(rng, n=60):
    dead = np.where(alive.to_numpy() == 0)[0]
    if not len(dead): return
    idx = rng.choice(dead, size=min(n, len(dead)), replace=False)
    r_f = r_pos.to_numpy(); ph_f = ph_pos.to_numpy()
    rd_f = r_dot.to_numpy(); lz_f = ang_mom.to_numpy(); al_f = alive.to_numpy()
    u = rng.uniform(0.0, 1.0, len(idx))
    r_n = R_ISCO * (R_OUT / R_ISCO)**u
    ph_n = rng.uniform(0.0, 2.0 * np.pi, len(idx))
    for j, i in enumerate(idx):
        ri = r_n[j]; d = max(ri - 3.0 * M, 0.10)
        r_f[i] = ri; ph_f[i] = ph_n[j]; rd_f[i] = -abs(rng.normal(0.0, 0.01))
        lz_f[i] = math.sqrt(M * ri * ri / d) * (0.98 + rng.normal(0.0, 0.015))
        al_f[i] = 1
    r_pos.from_numpy(r_f); ph_pos.from_numpy(ph_f)
    r_dot.from_numpy(rd_f); ang_mom.from_numpy(lz_f); alive.from_numpy(al_f)

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    global cam_theta, cam_phi, cam_radius
    rng = np.random.default_rng(7)
    update_camera()
    init_particles()
    init_stars()

    gui = ti.GUI(
        "Schwarzschild BH — 3D Interactive Lensing  [WASD/Arrows: Orbit, Q/E: Zoom]",
        res=(WIN_W, WIN_H), fast_gui=True,
    )

    print()
    print("Controls: WASD or Arrow Keys to Orbit, Q/E or Space/Shift to Zoom")
    print()

    frame = 0; t0 = time.perf_counter()

    while gui.running:
        for e in gui.get_events():
            if e.key == gui.ESCAPE:
                gui.running = False
        
        # Camera controls
        if gui.is_pressed(gui.LEFT, 'a'): cam_phi -= 0.05
        if gui.is_pressed(gui.RIGHT, 'd'): cam_phi += 0.05
        if gui.is_pressed(gui.UP, 'w'): cam_theta -= 0.05
        if gui.is_pressed(gui.DOWN, 's'): cam_theta += 0.05
        if gui.is_pressed(gui.SPACE, 'q'): cam_radius *= 0.95
        if gui.is_pressed(gui.SHIFT, 'e'): cam_radius *= 1.05

        update_camera()
        update_particles()
        decay_fb()
        render_particles()
        render_stars_lensed()
        render_horizon()
        tonemap()

        gui.set_image(fb)
        gui.show()

        if frame % 50 == 0 and frame > 0:
            respawn_dead(rng, n=60)
        
        frame += 1

if __name__ == "__main__":
    main()
