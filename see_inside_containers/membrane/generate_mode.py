# generate_mode.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib import colors
from mpl_toolkits.mplot3d import proj3d
import argparse
from datetime import datetime

# ----------------------- utilities -----------------------
def _proj_pixels(ax, X3, Y3, Z3):
    """Project 3D arrays to display-pixel coordinates."""
    M = ax.get_proj()
    x2, y2, _ = proj3d.proj_transform(X3.ravel(), Y3.ravel(), Z3.ravel(), M)
    xy_disp = ax.transData.transform(np.c_[x2, y2])
    Xp = xy_disp[:, 0].reshape(X3.shape)
    Yp = xy_disp[:, 1].reshape(X3.shape)
    return Xp, Yp

def _pixel_scales(ax, X, Z, U, x, z):
    """
    Pixels per unit of x or z at each surface point, for thickness compensation.
    """
    X3, Y3, Z3 = U, Z, -X
    # one draw to ensure transforms are current
    ax.figure.canvas.draw()
    Xp, Yp = _proj_pixels(ax, X3, Y3, Z3)

    # derivatives in pixels per unit parameter
    dXp_dx = np.gradient(Xp, x, axis=1, edge_order=2)
    dYp_dx = np.gradient(Yp, x, axis=1, edge_order=2)
    dXp_dz = np.gradient(Xp, z, axis=0, edge_order=2)
    dYp_dz = np.gradient(Yp, z, axis=0, edge_order=2)

    Sx = np.hypot(dXp_dx, dYp_dx)  # px / unit-x
    Sz = np.hypot(dXp_dz, dYp_dz)  # px / unit-z
    return Sx, Sz

def _grid_alpha_px(ax, X, Z, U, x, z, Lx, Lz, grid_nx, grid_nz,
                   width_px=2.0, border_px=2.0):
    """
    Pixel-correct grid mask (alpha in [0,1]) with constant on-screen thickness.
    """
    # --- Constants ---
    ramp_px = 1.0  # Use a 1px ramp for antialiasing
    eps = 1e-9

    # --- Pixel scales ---
    Sx, Sz = _pixel_scales(ax, X, Z, U, x, z)  # px per unit-x / unit-z

    # --- Parameter-space setup ---
    sx = (X / Lx) * grid_nx  # param-space x in "cell" units
    sz = (Z / Lz) * grid_nz  # param-space z in "cell" units
    cell_x = Lx / grid_nx
    cell_z = Lz / grid_nz

    # --- Grid line alpha ---
    dx_cell = np.abs(sx - np.round(sx))
    dz_cell = np.abs(sz - np.round(sz))

    half_width_px = width_px / 2.0
    # Convert pixel widths to cell units for vertical lines
    hwx_cell = half_width_px / (Sx + eps) / cell_x
    rwx_cell = ramp_px / (Sx + eps) / cell_x
    alpha_x = np.clip((hwx_cell - dx_cell) / (rwx_cell + eps), 0.0, 1.0)

    # Convert pixel widths to cell units for horizontal lines
    hwz_cell = half_width_px / (Sz + eps) / cell_z
    rwz_cell = ramp_px / (Sz + eps) / cell_z
    alpha_z = np.clip((hwz_cell - dz_cell) / (rwz_cell + eps), 0.0, 1.0)

    alpha_grid = 1.0 - (1.0 - alpha_x) * (1.0 - alpha_z)

    # --- Border alpha ---
    # Calculate distance to nearest edge in pixels for a uniform border
    d_edge_x_cells = np.minimum(sx, grid_nx - sx)
    d_edge_z_cells = np.minimum(sz, grid_nz - sz)

    d_edge_px_x = d_edge_x_cells * cell_x * Sx
    d_edge_px_z = d_edge_z_cells * cell_z * Sz

    d_edge_px = np.minimum(d_edge_px_x, d_edge_px_z)

    # Create a sharp border with a 1px antialiased inner edge
    alpha_border = np.clip((border_px - d_edge_px) / ramp_px, 0.0, 1.0)

    return np.maximum(alpha_grid, alpha_border)

# ----------------------- main renderer -----------------------
def mode(m, n, c, amplitude, ratio, grid_nx, grid_nz, width_px, border_px, preview=False):
    """
    Vibrating rectangular membrane with pixel-correct grid baked into facecolors.
    """
    # geometry
    Lz = 1.0
    Lx = ratio * Lz

    fps = 20
    nz_surf, nx_surf = 600, int(600 * ratio)  # denser grid reduces aliasing
    x = np.linspace(0.0, Lx, nx_surf)
    z = np.linspace(0.0, Lz, nz_surf)
    X, Z = np.meshgrid(x, z, indexing="xy")

    Psi = np.sin(m * np.pi * X / Lx) * np.sin(n * np.pi * Z / Lz)
    f_mn = 0.5 * c * np.sqrt((m / Lx) ** 2 + (n / Lz) ** 2)
    T = np.inf if f_mn == 0 else 1.0 / f_mn
    nframes = fps if not np.isfinite(T) else max(1, int(round(2 * T * fps)))
    t_step = 1.0 / fps

    # figure
    fig = plt.figure(figsize=(8, 8), dpi=200)  # higher DPI => steadier lines
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_position([0, 0, 1, 1])
    ax.set_axis_off()

    # colormap
    cmap = plt.get_cmap("PiYG")
    rng = np.random.default_rng(0)
    dither = (rng.random(Psi.shape) - 0.5) * (1.0 / 255.0)

    def colorize(U, use_dither=False):
        V = U + (dither if use_dither else 0.0)
        # ------------------- FIX -------------------
        # Normalize based on the overall max amplitude for a consistent colormap
        vmax = amplitude or 1e-9
        # -------------------------------------------
        norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        return cmap(norm(V))

    # view
    ax.view_init(elev=15, azim=10)
    max_extent = max(amplitude, 0.01) # Avoid zero extent for box aspect
    ax.set_xlim(-max_extent * 1.25, max_extent * 1.25)
    ax.set_ylim(0.0, Lz)
    ax.set_zlim(-Lx, 0.0)
    try:
        ax.set_box_aspect((2.0 * max_extent * 1.25, Lz, Lx), zoom=1)
    except TypeError:
        # Fallback for older Matplotlib versions
        ax.set_box_aspect((2.0 * max_extent * 1.25, Lz, Lx))
        ax.dist = 6

    # helpers
    def facecolors_with_grid(U):
        # pixel-correct grid alpha
        a = _grid_alpha_px(ax, X, Z, U, x, z, Lx, Lz, grid_nx, grid_nz,
                           width_px=width_px, border_px=border_px)
        fc = colorize(U, use_dither=False)
        # blend toward black where grid alpha > 0
        fc[..., :3] = (1.0 - a)[..., None] * fc[..., :3]
        return fc

    # initial surface
    U0 = amplitude * Psi
    surf = ax.plot_surface(
        U0, Z, -X,
        facecolors=facecolors_with_grid(U0),
        rcount=Z.shape[0], ccount=Z.shape[1],
        linewidth=0, edgecolor="none", antialiased=False, shade=False
    )

    # animation
    def update(k, use_dither=False):
        nonlocal surf
        phase = np.cos(2 * np.pi * f_mn * (k * t_step))
        U = amplitude * Psi * phase
        surf.remove()
        surf = ax.plot_surface(
            U, Z, -X,
            facecolors=facecolors_with_grid(U),
            rcount=Z.shape[0], ccount=Z.shape[1],
            linewidth=0, edgecolor="none", antialiased=False, shade=False
        )
        return [surf]

    # output
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if preview:
        update(0)
        fn = f"preview_m{m}n{n}_{ts}.png"
        plt.savefig(fn, dpi=fig.dpi, pad_inches=0)
        print(f"Saved: {fn}")
    else:
        fn = f"mode_m{m}n{n}_{ts}.mp4"
        anim = FuncAnimation(fig, update, fargs=(False,), frames=max(1, nframes // 2),
                             interval=1000 / fps, blit=False)
        writer = FFMpegWriter(fps=fps, codec="h264", extra_args=["-g", "1", "-pix_fmt", "yuv420p"])
        anim.save(fn, writer=writer, dpi=fig.dpi, savefig_kwargs={"pad_inches": 0})
        print(f"Saved: {fn}")
    plt.close(fig)

# ----------------------- CLI -----------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Membrane with pixel-constant grid baked into facecolors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("-m", type=int, required=True, help="Mode number for x-dimension.")
    ap.add_argument("-n", type=int, required=True, help="Mode number for z-dimension.")
    ap.add_argument("-c", type=float, default=1.0, help="Wave speed.")
    # ------------------- UPDATED DEFAULTS -------------------
    ap.add_argument("-a", "--amplitude", type=float, default=0.8, help="Vibration amplitude.")
    ap.add_argument("--grid_nx", type=int, default=14, help="Number of grid cells in x.")
    ap.add_argument("--grid_nz", type=int, default=7, help="Number of grid cells in z.")
    ap.add_argument("--grid_px", type=float, default=10, help="Grid line thickness in pixels.")
    ap.add_argument("--border_px", type=float, default=10, help="Border thickness in pixels.")
    # ---------------------------------------------------------
    ap.add_argument("-r", "--ratio", type=float, default=2.5, help="Aspect ratio (Lx / Lz).")
    ap.add_argument("--preview", action="store_true", help="Generate a static PNG instead of an MP4.")

    args = ap.parse_args()
    mode(
        m=args.m,
        n=args.n,
        c=args.c,
        amplitude=args.amplitude,
        ratio=args.ratio,
        grid_nx=args.grid_nx,
        grid_nz=args.grid_nz,
        width_px=args.grid_px,
        border_px=args.border_px,
        preview=args.preview,
    )
