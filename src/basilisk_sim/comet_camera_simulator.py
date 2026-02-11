"""
Comet Camera Simulator - Visualizes Vibration-Induced Blur in Long Exposures.

This module renders synthetic comet images using attitude jitter from simulation
outputs. It is designed to pair with vizard_demo.py and show the practical
impact of input shaping on image quality.

Key idea:
- Small attitude jitter maps to large pixel blur for narrow field-of-view (FOV)
  telescopes at long range.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Unit conversion constants.
ARCSEC_PER_RAD = 180.0 / np.pi * 3600.0


class CometCameraSimulator:
    """
    Simulate comet images with optional motion blur from attitude jitter.

    Inputs:
    - fov_deg: camera field of view in degrees (assumes square FOV).
    - resolution: image resolution as (width_px, height_px).
    - pixel_scale: arcsec/pixel. If None, computed from fov_deg and resolution.
    - exposure_time: exposure duration in seconds.
    - noise_level: additive Gaussian noise standard deviation in [0, 1].

    Outputs:
    - generate_image returns a synthetic image and blur statistics.

    Process:
    - Convert attitude jitter (MRP) into angular displacements in arcsec.
    - Convert angular jitter into pixel offsets.
    - Render a comet nucleus/coma/tail at multiple offsets to emulate blur.
    - Add background stars and noise, then clip to [0, 1].
    """

    def __init__(
        self,
        fov_deg: float = 0.5,
        resolution: Tuple[int, int] = (512, 512),
        pixel_scale: Optional[float] = None,
        exposure_time: float = 2.0,
        noise_level: float = 0.01,
    ) -> None:
        # Camera configuration.
        self.fov_deg = float(fov_deg)
        self.resolution = tuple(resolution)
        self.exposure_time = float(exposure_time)
        self.noise_level = float(noise_level)

        # Pixel scale (arcsec per pixel). Use horizontal resolution for square FOV.
        if pixel_scale is None:
            self.pixel_scale = (self.fov_deg * 3600.0) / self.resolution[0]
        else:
            self.pixel_scale = float(pixel_scale)

        print("Comet Camera Configuration:")
        print(f"  FOV: {self.fov_deg} deg x {self.fov_deg} deg (telephoto)")
        print(f"  Resolution: {self.resolution[0]} x {self.resolution[1]} pixels")
        print(f"  Pixel scale: {self.pixel_scale:.2f} arcsec/pixel")
        print(f"  Exposure time: {self.exposure_time:.2f} s")

        # Comet geometry parameters (in pixels).
        self.comet_center = (self.resolution[0] // 2, self.resolution[1] // 2)
        self.comet_nucleus_radius = 8
        self.comet_coma_radius = 40
        self.comet_tail_length = 150
        self.comet_tail_angle = np.radians(45.0)

        # Background star field (fixed for repeatability).
        rng = np.random.default_rng(42)
        self.num_stars = 20
        self.star_positions = rng.random((self.num_stars, 2)) * np.array(
            [self.resolution[0], self.resolution[1]]
        )
        self.star_brightness = rng.uniform(0.1, 0.3, self.num_stars)

    def _angular_to_pixels(self, angle_arcsec: np.ndarray) -> np.ndarray:
        """Convert angular displacement in arcsec to pixel displacement."""
        return angle_arcsec / self.pixel_scale

    def _compute_attitude_jitter(
        self,
        exposure_start: float,
        sigma_history: np.ndarray,
        time_history: np.ndarray,
    ) -> np.ndarray:
        """
        Extract attitude jitter during the exposure window.

        Inputs:
        - exposure_start: start time of the exposure in seconds.
        - sigma_history: attitude history as MRPs (N x 3).
        - time_history: corresponding time stamps (N,).

        Output:
        - Array of (dx_arcsec, dy_arcsec) offsets for each sample.

        Process:
        - Select samples within [exposure_start, exposure_start + exposure_time].
        - Subtract mean MRPs to isolate jitter about the mean pointing.
        - Convert small-angle MRP deviations to arcsec using angle ~= 4*sigma.
        """
        exposure_mask = (time_history >= exposure_start) & (
            time_history <= exposure_start + self.exposure_time
        )

        if not np.any(exposure_mask):
            return np.zeros((1, 2))

        sigma_exposure = sigma_history[exposure_mask]
        sigma_mean = np.mean(sigma_exposure, axis=0)

        # Convert small angle MRPs to arcsec in the image plane.
        angular_disp = []
        for sigma in sigma_exposure:
            delta_sigma = sigma - sigma_mean
            angle_x = 4.0 * delta_sigma[0] * ARCSEC_PER_RAD
            angle_y = 4.0 * delta_sigma[1] * ARCSEC_PER_RAD
            angular_disp.append([angle_x, angle_y])

        return np.array(angular_disp)

    def _render_comet(
        self, image: np.ndarray, blur_trail: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Render comet nucleus, coma, and tail into an image.

        Inputs:
        - image: target image buffer (modified in-place).
        - blur_trail: list of (dx_px, dy_px) offsets. If None, render sharp.

        Output:
        - The same image array with comet features added.
        """
        h, w = image.shape
        cx, cy = self.comet_center
        y, x = np.ogrid[:h, :w]

        # Helper for a single comet render at a shifted center.
        def add_comet_at(center_x: float, center_y: float, weight: float) -> None:
            r_nucleus = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

            # Bright nucleus core.
            nucleus = weight * 1.0 * np.exp(
                -r_nucleus**2 / (2 * self.comet_nucleus_radius**2)
            )
            # Diffuse coma.
            coma = weight * 0.4 * np.exp(-r_nucleus / self.comet_coma_radius)

            # Tail in rotated coordinates.
            tail_dx = x - center_x
            tail_dy = y - center_y
            tail_x = tail_dx * np.cos(self.comet_tail_angle) + tail_dy * np.sin(
                self.comet_tail_angle
            )
            tail_y = -tail_dx * np.sin(self.comet_tail_angle) + tail_dy * np.cos(
                self.comet_tail_angle
            )

            tail_mask = tail_x < 0
            tail_distance = np.abs(tail_x)
            tail_width = 20 + 0.3 * tail_distance

            tail = np.zeros_like(image)
            tail[tail_mask] = (
                weight
                * 0.3
                * np.exp(-tail_distance[tail_mask] / self.comet_tail_length)
                * np.exp(-tail_y[tail_mask] ** 2 / (2 * tail_width[tail_mask] ** 2))
            )

            image[:] = image + nucleus + coma + tail

        if blur_trail is None or len(blur_trail) < 2:
            add_comet_at(cx, cy, 1.0)
        else:
            n_samples = len(blur_trail)
            for dx, dy in blur_trail:
                add_comet_at(cx + dx, cy + dy, 1.0 / n_samples)

        return image

    def _render_background_stars(
        self, image: np.ndarray, blur_trail: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Render background stars with optional motion blur streaks.

        Inputs:
        - image: target image buffer (modified in-place).
        - blur_trail: list of (dx_px, dy_px) offsets. If None, render points.
        """
        h, w = image.shape

        for i in range(self.num_stars):
            x_star, y_star = self.star_positions[i]
            brightness = self.star_brightness[i]

            if blur_trail is None or len(blur_trail) < 2:
                # Point source PSF.
                sigma_psf = 1.2
                y_grid, x_grid = np.ogrid[
                    max(0, int(y_star) - 8) : min(h, int(y_star) + 8),
                    max(0, int(x_star) - 8) : min(w, int(x_star) + 8),
                ]

                if y_grid.size > 0 and x_grid.size > 0:
                    psf = np.exp(
                        -((x_grid - x_star) ** 2 + (y_grid - y_star) ** 2)
                        / (2 * sigma_psf**2)
                    )
                    image[
                        max(0, int(y_star) - 8) : min(h, int(y_star) + 8),
                        max(0, int(x_star) - 8) : min(w, int(x_star) + 8),
                    ] += brightness * psf
            else:
                # Motion blurred streaks (subsample for efficiency).
                n_samples = min(len(blur_trail), 50)
                step = max(1, len(blur_trail) // n_samples)

                for dx, dy in blur_trail[::step]:
                    star_x = x_star + dx
                    star_y = y_star + dy

                    if 0 <= star_x < w and 0 <= star_y < h:
                        sigma_psf = 1.2
                        y_grid, x_grid = np.ogrid[
                            max(0, int(star_y) - 4) : min(h, int(star_y) + 4),
                            max(0, int(star_x) - 4) : min(w, int(star_x) + 4),
                        ]

                        if y_grid.size > 0 and x_grid.size > 0:
                            psf = np.exp(
                                -((x_grid - star_x) ** 2 + (y_grid - star_y) ** 2)
                                / (2 * sigma_psf**2)
                            )
                            image[
                                max(0, int(star_y) - 4) : min(h, int(star_y) + 4),
                                max(0, int(star_x) - 4) : min(w, int(star_x) + 4),
                            ] += (brightness / n_samples) * psf

        return image

    def generate_image(
        self,
        sigma_history: Optional[np.ndarray] = None,
        time_history: Optional[np.ndarray] = None,
        exposure_start: float = 0.0,
        jitter_arcsec_history: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Render a comet image with optional motion blur from attitude data.

        Inputs:
        - sigma_history: MRP history (N x 3). If None, render a sharp image.
        - time_history: time stamps (N,). Required if sigma_history is provided.
        - exposure_start: start time of the exposure.
        - jitter_arcsec_history: optional (N x 2) angular jitter directly in arcsec.
          If provided, this takes precedence over sigma/time conversion.

        Outputs:
        - image: (H x W) float array in [0, 1].
        - stats: dictionary with RMS/peak jitter and blur in pixels.
        """
        image = np.zeros(self.resolution)

        blur_trail = None
        blur_stats = {"rms_arcsec": 0.0, "peak_arcsec": 0.0, "blur_pixels": 0.0}

        if jitter_arcsec_history is not None:
            jitter = np.asarray(jitter_arcsec_history, dtype=float)
            if jitter.ndim == 2 and jitter.shape[1] == 2 and len(jitter) > 1:
                blur_trail = self._angular_to_pixels(jitter)
                jitter_magnitude = np.sqrt(jitter[:, 0] ** 2 + jitter[:, 1] ** 2)
                blur_stats["rms_arcsec"] = float(np.sqrt(np.mean(jitter_magnitude**2)))
                blur_stats["peak_arcsec"] = float(np.max(jitter_magnitude))
                blur_stats["blur_pixels"] = float(
                    self._angular_to_pixels(blur_stats["rms_arcsec"])
                )
        elif sigma_history is not None and time_history is not None:
            jitter = self._compute_attitude_jitter(
                exposure_start, sigma_history, time_history
            )
            if len(jitter) > 1:
                blur_trail = self._angular_to_pixels(jitter)

                jitter_magnitude = np.sqrt(jitter[:, 0] ** 2 + jitter[:, 1] ** 2)
                blur_stats["rms_arcsec"] = float(np.sqrt(np.mean(jitter_magnitude**2)))
                blur_stats["peak_arcsec"] = float(np.max(jitter_magnitude))
                blur_stats["blur_pixels"] = float(
                    self._angular_to_pixels(blur_stats["rms_arcsec"])
                )

        # Render comet and stars, then add noise.
        image = self._render_comet(image, blur_trail)
        image = self._render_background_stars(image, blur_trail)

        image += np.random.normal(0.0, self.noise_level, self.resolution)
        image = np.clip(image, 0.0, 1.0)

        return image, blur_stats


def load_simulation_data(filename: str) -> Dict[str, Optional[np.ndarray]]:
    """Load simulation results stored in an NPZ file."""
    data = np.load(filename, allow_pickle=True)
    return {
        "time": data["time"],
        "sigma": data["sigma"],
        "mode1": data["mode1"] if "mode1" in data else None,
        "mode2": data["mode2"] if "mode2" in data else None,
        "method": str(data["method"]),
    }


def _extract_modal_jitter_arcsec(
    time_history: np.ndarray,
    mode1: np.ndarray,
    mode2: np.ndarray,
    exposure_start: float,
    exposure_time: float,
    lever_arm_m: float = 4.0,
    modal_to_pointing_gain: float = 1.0,
) -> np.ndarray:
    """
    Convert modal displacements to angular jitter over one exposure window.

    Mean offsets inside the window are removed so static bias is not treated
    as motion blur.
    """
    mask = (time_history >= exposure_start) & (time_history <= exposure_start + exposure_time)
    if not np.any(mask):
        return np.zeros((1, 2))

    mode1_exp = np.asarray(mode1[mask], dtype=float)
    mode2_exp = np.asarray(mode2[mask], dtype=float)
    if mode1_exp.size < 2 or mode2_exp.size < 2:
        return np.zeros((1, 2))

    mode1_centered = mode1_exp - np.mean(mode1_exp)
    mode2_centered = mode2_exp - np.mean(mode2_exp)
    scale = float(modal_to_pointing_gain) * ARCSEC_PER_RAD / max(float(lever_arm_m), 1e-9)
    return np.column_stack((mode1_centered * scale, mode2_centered * scale))


def render_camera_sidecar_frames_from_npz(
    npz_file: str,
    output_dir: str,
    *,
    fps: float = 5.0,
    exposure_time: float = 2.0,
    fov_deg: float = 0.5,
    resolution: Tuple[int, int] = (512, 512),
    noise_level: float = 0.01,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    prefix: str = "camera",
    prefer_modal_jitter: bool = True,
    lever_arm_m: float = 4.0,
    modal_to_pointing_gain: float = 1.0,
) -> Dict[str, str]:
    """
    Render a synchronized camera sidecar frame sequence from mission NPZ data.

    This does not modify Vizard's internal camera rendering. It produces
    frame-accurate synthetic sensor images aligned to the saved attitude history.

    Returns:
    - Dict containing paths to frame directory and metadata CSV.
    """
    data = np.load(npz_file, allow_pickle=True)
    time_history = np.asarray(data["time"], dtype=float)
    sigma_history = np.asarray(data["sigma"], dtype=float)
    if sigma_history.ndim != 2 or sigma_history.shape[1] != 3:
        raise ValueError(f"Invalid sigma shape in {npz_file}: {sigma_history.shape}")

    if len(time_history) < 2:
        raise ValueError(f"Not enough time samples in {npz_file} to render frames.")

    mode1 = np.asarray(data["mode1"], dtype=float) if "mode1" in data.files else None
    mode2 = np.asarray(data["mode2"], dtype=float) if "mode2" in data.files else None
    use_modal = (
        bool(prefer_modal_jitter)
        and mode1 is not None
        and mode2 is not None
        and mode1.shape == time_history.shape
        and mode2.shape == time_history.shape
    )

    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, f"{prefix}_frames")
    os.makedirs(frames_dir, exist_ok=True)

    t0 = float(time_history[0]) if start_time is None else float(start_time)
    tf = float(time_history[-1]) if end_time is None else float(end_time)
    if tf <= t0:
        raise ValueError(f"Invalid frame window [{t0}, {tf}] from {npz_file}")

    fps = max(float(fps), 0.1)
    exposure_time = max(float(exposure_time), 1e-3)
    frame_period = 1.0 / fps

    first_frame_time = max(t0 + exposure_time, float(time_history[0]) + exposure_time)
    frame_times = np.arange(first_frame_time, tf + 0.5 * frame_period, frame_period)
    if frame_times.size == 0:
        frame_times = np.array([min(tf, first_frame_time)], dtype=float)

    camera = CometCameraSimulator(
        fov_deg=float(fov_deg),
        resolution=tuple(resolution),
        exposure_time=exposure_time,
        noise_level=float(noise_level),
    )

    metadata_rows = []
    for i, t_frame in enumerate(frame_times):
        exposure_start = float(max(time_history[0], t_frame - exposure_time))
        if use_modal:
            jitter_arcsec = _extract_modal_jitter_arcsec(
                time_history=time_history,
                mode1=mode1,
                mode2=mode2,
                exposure_start=exposure_start,
                exposure_time=exposure_time,
                lever_arm_m=lever_arm_m,
                modal_to_pointing_gain=modal_to_pointing_gain,
            )
            image, stats = camera.generate_image(jitter_arcsec_history=jitter_arcsec)
        else:
            image, stats = camera.generate_image(
                sigma_history=sigma_history,
                time_history=time_history,
                exposure_start=exposure_start,
            )
        frame_name = f"{prefix}_{i:05d}.png"
        frame_path = os.path.join(frames_dir, frame_name)
        plt.imsave(frame_path, image, cmap="gray", vmin=0.0, vmax=1.0)
        metadata_rows.append(
            {
                "frame_idx": i,
                "time_s": float(t_frame),
                "exposure_start_s": exposure_start,
                "exposure_time_s": exposure_time,
                "rms_arcsec": float(stats["rms_arcsec"]),
                "peak_arcsec": float(stats["peak_arcsec"]),
                "blur_pixels": float(stats["blur_pixels"]),
                "file": frame_name,
            }
        )

    csv_path = os.path.join(output_dir, f"{prefix}_metadata.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_idx",
                "time_s",
                "exposure_start_s",
                "exposure_time_s",
                "rms_arcsec",
                "peak_arcsec",
                "blur_pixels",
                "file",
            ],
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    summary = {
        "frames_dir": frames_dir,
        "metadata_csv": csv_path,
        "n_frames": str(len(metadata_rows)),
    }
    print(
        f"Saved camera sidecar sequence: {summary['n_frames']} frames -> {summary['frames_dir']}"
    )
    print(f"Saved camera sidecar metadata: {summary['metadata_csv']}")
    return summary


def create_comet_comparison_figure(
    s_curve_data: Dict[str, np.ndarray],
    fourth_data: Dict[str, np.ndarray],
    output_filename: str = "comet_blur_comparison.png",
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Create a side-by-side comparison of S-curve vs fourth-order comet images.

    Inputs:
    - s_curve_data: dict containing time and sigma arrays.
    - fourth_data: dict containing time and sigma arrays.
    - output_filename: output path for the PNG figure.

    Outputs:
    - Tuple of (S-curve blur stats, fourth-order blur stats).
    """
    camera = CometCameraSimulator(
        fov_deg=0.5,
        resolution=(512, 512),
        exposure_time=2.0,
        noise_level=0.01,
    )

    # Expose after the slew completes.
    slew_end_time = 30.0
    exposure_start = slew_end_time + 1.0

    print(f"\nGenerating comet images (exposure at t={exposure_start:.1f}s)...")

    img_s_curve, stats_s_curve = camera.generate_image(
        s_curve_data["sigma"], s_curve_data["time"], exposure_start
    )
    img_fourth, stats_fourth = camera.generate_image(
        fourth_data["sigma"], fourth_data["time"], exposure_start
    )

    # Reference image without jitter.
    img_perfect, _ = camera.generate_image()

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    cmap = plt.cm.afmhot

    axes[0].imshow(img_perfect, cmap=cmap, origin="lower", vmin=0, vmax=0.9)
    axes[0].set_title("Reference\n(Perfect Pointing)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Sharp comet nucleus and tail")
    axes[0].axis("off")
    axes[0].text(
        0.02,
        0.98,
        f"FOV: {camera.fov_deg} deg",
        transform=axes[0].transAxes,
        fontsize=9,
        verticalalignment="top",
        color="white",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.5),
    )

    axes[1].imshow(img_s_curve, cmap=cmap, origin="lower", vmin=0, vmax=0.9)
    axes[1].set_title(
        "S-Curve Reference",
        fontsize=12,
        fontweight="bold",
        color="orange",
    )
    blur_text = (
        f"Blur: {stats_s_curve['blur_pixels']:.1f} px\n"
        f"({stats_s_curve['rms_arcsec']:.1f} arcsec RMS)"
    )
    axes[1].set_xlabel(blur_text, fontsize=10, color="orange")
    axes[1].axis("off")

    if stats_s_curve["blur_pixels"] > 5:
        axes[1].text(
            0.5,
            0.02,
            "UNUSABLE",
            transform=axes[1].transAxes,
            fontsize=14,
            ha="center",
            color="orange",
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    axes[2].imshow(img_fourth, cmap=cmap, origin="lower", vmin=0, vmax=0.9)
    method_name = (
        fourth_data["method"].upper()
        if isinstance(fourth_data["method"], str)
        else "SHAPED"
    )
    axes[2].set_title(
        f"{method_name} Input Shaping",
        fontsize=12,
        fontweight="bold",
        color="green",
    )
    blur_text = (
        f"Blur: {stats_fourth['blur_pixels']:.1f} px\n"
        f"({stats_fourth['rms_arcsec']:.1f} arcsec RMS)"
    )
    axes[2].set_xlabel(blur_text, fontsize=10, color="green")
    axes[2].axis("off")

    if stats_fourth["blur_pixels"] < 2:
        axes[2].text(
            0.5,
            0.02,
            "SCIENCE QUALITY",
            transform=axes[2].transAxes,
            fontsize=14,
            ha="center",
            color="green",
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    fig.suptitle(
        "Impact of Input Shaping on Comet Photography\n"
        "(2-second exposure during post-slew settling)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    if stats_s_curve["blur_pixels"] > 0.1:
        improvement = stats_s_curve["blur_pixels"] / max(
            stats_fourth["blur_pixels"], 0.1
        )
        fig.text(
            0.5,
            -0.02,
            f"Input shaping reduces motion blur by {improvement:.0f}x",
            ha="center",
            fontsize=12,
            style="italic",
        )

    plt.tight_layout()
    plt.savefig(output_filename, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {output_filename}")
    plt.close()

    return stats_s_curve, stats_fourth


def create_blur_demo_from_modal_data(
    s_curve_file: str,
    fourth_file: str,
    output_filename: str = "comet_blur_comparison.png",
) -> None:
    """
    Load modal displacement data and render a blur comparison.

    Inputs:
    - s_curve_file: NPZ with modal displacement data for S-curve case.
    - fourth_file: NPZ with modal displacement data for fourth-order case.
    - output_filename: output PNG file name.
    """
    s_curve = np.load(s_curve_file, allow_pickle=True)
    fourth = np.load(fourth_file, allow_pickle=True)

    time_u = np.asarray(s_curve["time"], dtype=float)
    time_s = np.asarray(fourth["time"], dtype=float)

    mode1_u = np.asarray(s_curve["mode1"], dtype=float) if "mode1" in s_curve else np.zeros_like(time_u)
    mode2_u = np.asarray(s_curve["mode2"], dtype=float) if "mode2" in s_curve else np.zeros_like(time_u)
    mode1_s = np.asarray(fourth["mode1"], dtype=float) if "mode1" in fourth else np.zeros_like(time_s)
    mode2_s = np.asarray(fourth["mode2"], dtype=float) if "mode2" in fourth else np.zeros_like(time_s)

    arm_length = 4.0
    jitter_x_u_arcsec = (mode1_u / arm_length) * ARCSEC_PER_RAD
    jitter_y_u_arcsec = (mode2_u / arm_length) * ARCSEC_PER_RAD
    jitter_x_s_arcsec = (mode1_s / arm_length) * ARCSEC_PER_RAD
    jitter_y_s_arcsec = (mode2_s / arm_length) * ARCSEC_PER_RAD
    jitter_u_arcsec = np.sqrt(jitter_x_u_arcsec**2 + jitter_y_u_arcsec**2)
    jitter_s_arcsec = np.sqrt(jitter_x_s_arcsec**2 + jitter_y_s_arcsec**2)
    pixel_scale = (0.5 * 3600.0) / 512.0

    slew_end = 30.0
    post_mask_u = time_u >= slew_end
    post_mask_s = time_s >= slew_end
    post_slew_u_mag = jitter_u_arcsec[post_mask_u]
    post_slew_s_mag = jitter_s_arcsec[post_mask_s]
    rms_u = np.sqrt(np.mean(post_slew_u_mag**2)) if post_slew_u_mag.size > 0 else 0.0
    rms_s = np.sqrt(np.mean(post_slew_s_mag**2)) if post_slew_s_mag.size > 0 else 0.0
    peak_u = np.max(post_slew_u_mag) if post_slew_u_mag.size > 0 else 0.0
    peak_s = np.max(post_slew_s_mag) if post_slew_s_mag.size > 0 else 0.0
    blur_px_u = rms_u / pixel_scale
    blur_px_s = rms_s / pixel_scale

    # Deterministic exposure window rendering from modal trails.
    exposure_start = 31.0
    exposure_time = 2.0
    jitter_exp_u = _extract_modal_jitter_arcsec(
        time_history=time_u,
        mode1=mode1_u,
        mode2=mode2_u,
        exposure_start=exposure_start,
        exposure_time=exposure_time,
        lever_arm_m=arm_length,
        modal_to_pointing_gain=1.0,
    )
    jitter_exp_s = _extract_modal_jitter_arcsec(
        time_history=time_s,
        mode1=mode1_s,
        mode2=mode2_s,
        exposure_start=exposure_start,
        exposure_time=exposure_time,
        lever_arm_m=arm_length,
        modal_to_pointing_gain=1.0,
    )

    camera = CometCameraSimulator(
        fov_deg=0.5,
        resolution=(512, 512),
        exposure_time=exposure_time,
        noise_level=0.01,
    )
    img_ref, _ = camera.generate_image()
    img_u, img_stats_u = camera.generate_image(jitter_arcsec_history=jitter_exp_u)
    img_s, img_stats_s = camera.generate_image(jitter_arcsec_history=jitter_exp_s)

    print("\n" + "=" * 60)
    print("COMET PHOTOGRAPHY MOTION BLUR ANALYSIS")
    print("=" * 60)
    print("\nS-Curve:")
    print(f"  Angular jitter RMS:  {rms_u:.1f} arcsec")
    print(f"  Angular jitter Peak: {peak_u:.1f} arcsec")
    print(f"  Image blur (RMS):    {blur_px_u:.1f} pixels")
    print("\nFourth-Order Shaped:")
    print(f"  Angular jitter RMS:  {rms_s:.1f} arcsec")
    print(f"  Angular jitter Peak: {peak_s:.1f} arcsec")
    print(f"  Image blur (RMS):    {blur_px_s:.1f} pixels")

    if rms_s > 1e-9:
        print(f"\nImprovement: {rms_u / rms_s:.1f}x reduction in RMS jitter")

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.2])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time_u, jitter_u_arcsec, "r-", alpha=0.7, linewidth=0.5)
    ax1.axvline(x=30, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Angular Jitter (arcsec)")
    ax1.set_title("S-curve: Full Timeline", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 60])

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time_s, jitter_s_arcsec, "g-", alpha=0.7, linewidth=0.5)
    ax2.axvline(x=30, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Angular Jitter (arcsec)")
    ax2.set_title("Fourth-order: Full Timeline", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 60])

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(time_u[time_u >= 30], jitter_u_arcsec[time_u >= 30], "r-", alpha=0.7, label="S-curve", linewidth=0.8)
    ax3.plot(time_s[time_s >= 30], jitter_s_arcsec[time_s >= 30], "g-", alpha=0.7, label="Fourth-order", linewidth=0.8)
    ax3.axhline(y=pixel_scale, color="blue", linestyle=":", label="1 pixel", alpha=0.5)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Angular Jitter (arcsec)")
    ax3.set_title("Post-Slew Comparison", fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, :])
    from scipy import signal as sig

    data_ux = jitter_x_u_arcsec[post_mask_u]
    data_uy = jitter_y_u_arcsec[post_mask_u]
    data_sx = jitter_x_s_arcsec[post_mask_s]
    data_sy = jitter_y_s_arcsec[post_mask_s]
    if min(len(data_ux), len(data_uy), len(data_sx), len(data_sy)) > 256:
        # Keep signed components; summing axis PSDs avoids magnitude rectification artifacts.
        data_ux = data_ux - np.mean(data_ux)
        data_uy = data_uy - np.mean(data_uy)
        data_sx = data_sx - np.mean(data_sx)
        data_sy = data_sy - np.mean(data_sy)

        dt_u = float(np.median(np.diff(time_u)))
        dt_s = float(np.median(np.diff(time_s)))
        fs_u = 1.0 / dt_u
        fs_s = 1.0 / dt_s

        # Use dense zero padding so narrow modal peaks are visually resolved.
        base_u = 1 << int(np.ceil(np.log2(len(data_ux))))
        base_s = 1 << int(np.ceil(np.log2(len(data_sx))))
        nfft_u = min(1 << 20, max(65536, 16 * base_u))
        nfft_s = min(1 << 20, max(65536, 16 * base_s))

        f_u, psd_ux = sig.periodogram(
            data_ux, fs_u, window="hann", detrend=False, scaling="density", nfft=nfft_u
        )
        _, psd_uy = sig.periodogram(
            data_uy, fs_u, window="hann", detrend=False, scaling="density", nfft=nfft_u
        )
        f_s, psd_sx = sig.periodogram(
            data_sx, fs_s, window="hann", detrend=False, scaling="density", nfft=nfft_s
        )
        _, psd_sy = sig.periodogram(
            data_sy, fs_s, window="hann", detrend=False, scaling="density", nfft=nfft_s
        )
        psd_u = psd_ux + psd_uy
        psd_s = psd_sx + psd_sy

        ax4.semilogy(f_u, psd_u, "r-", linewidth=1.5, label="S-curve", alpha=0.9)
        ax4.semilogy(f_s, psd_s, "g-", linewidth=1.5, label="Fourth-order", alpha=0.9)
        for freq in [0.4, 1.3]:
            ax4.axvline(x=freq, color="blue", linestyle="--", alpha=0.5, linewidth=1)
        ax4.set_xlabel("Frequency (Hz)")
        ax4.set_ylabel("Power Spectral Density (arcsec^2/Hz)")
        ax4.set_title("Jitter Power Spectrum (Signed Axes)")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim([0, 1.5])

    ax5 = fig.add_subplot(gs[2, 0])
    ax5.imshow(img_ref, cmap="gray", origin="lower", vmin=0.0, vmax=0.85, interpolation="bilinear")
    ax5.set_title("Reference\n(No Motion)", fontsize=12, fontweight="bold")
    ax5.axis("off")

    ax6 = fig.add_subplot(gs[2, 1])
    ax6.imshow(img_u, cmap="gray", origin="lower", vmin=0.0, vmax=0.85, interpolation="bilinear")
    ax6.set_title(
        f"S-CURVE\nExposure RMS blur: {img_stats_u['blur_pixels']:.1f} px",
        fontsize=12,
        fontweight="bold",
        color="red",
    )
    ax6.axis("off")

    ax7 = fig.add_subplot(gs[2, 2])
    ax7.imshow(img_s, cmap="gray", origin="lower", vmin=0.0, vmax=0.85, interpolation="bilinear")
    ax7.set_title(
        f"FOURTH-ORDER\nExposure RMS blur: {img_stats_s['blur_pixels']:.1f} px",
        fontsize=12,
        fontweight="bold",
        color="green",
    )
    ax7.axis("off")

    fig.suptitle("Comet Mission Camera Simulator", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_filename, dpi=220, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {output_filename}")
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Comet Camera Simulator - Motion Blur Visualization")
    print("=" * 60)
    print("\nAligned with vizard_demo.py comet photography mission!")

    import argparse

    parser = argparse.ArgumentParser(
        description="Render comet blur comparison from matched mission NPZ files."
    )
    parser.add_argument(
        "--s-curve-file",
        default=None,
        help="Explicit path to S-curve NPZ.",
    )
    parser.add_argument(
        "--fourth-file",
        default=None,
        help="Explicit path to fourth-order NPZ.",
    )
    parser.add_argument(
        "--pair-dir",
        default=None,
        help="Directory containing both NPZ files. If omitted, output/cache is preferred.",
    )
    parser.add_argument(
        "--output",
        default="comet_blur_comparison.png",
        help="Output PNG file.",
    )
    args = parser.parse_args()

    s_curve_file = args.s_curve_file
    fourth_file = args.fourth_file

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))

    possible_s_curve = [
        "vizard_demo_s_curve_standard_pd.npz",
        "vizard_demo_s_curve.npz",
        "comparison_s_curve.npz",
    ]
    possible_fourth = [
        "vizard_demo_fourth_standard_pd.npz",
        "vizard_demo_fourth.npz",
        "comparison_fourth.npz",
    ]

    def _pick_file(base_dir: str, names: list[str]) -> Optional[str]:
        for name in names:
            path = os.path.join(base_dir, name)
            if os.path.exists(path):
                return path
        return None

    if s_curve_file is None or fourth_file is None:
        candidate_dirs = []
        if args.pair_dir:
            candidate_dirs.append(args.pair_dir)
        candidate_dirs.extend(
            [
                os.path.join(root_dir, "output", "cache"),
                os.getcwd(),
                root_dir,
                os.path.join(root_dir, "data", "trajectories"),
                os.path.join(root_dir, "output"),
            ]
        )

        # Require both files from the same directory to avoid mismatched comparisons.
        for base_dir in candidate_dirs:
            s_try = _pick_file(base_dir, possible_s_curve) if s_curve_file is None else s_curve_file
            f_try = _pick_file(base_dir, possible_fourth) if fourth_file is None else fourth_file
            if s_try and f_try and os.path.dirname(os.path.abspath(s_try)) == os.path.dirname(os.path.abspath(f_try)):
                s_curve_file = s_try
                fourth_file = f_try
                break

    if s_curve_file and fourth_file:
        print("\nUsing data files:")
        print(f"  S-curve: {s_curve_file}")
        print(f"  Fourth:  {fourth_file}")
        create_blur_demo_from_modal_data(
            s_curve_file, fourth_file, output_filename=args.output
        )
    else:
        print("\nNo simulation data found. Run run_vizard_demo.py first:")
        print("  python scripts/run_vizard_demo.py s_curve --controller standard_pd")
        print("  python scripts/run_vizard_demo.py fourth --controller standard_pd")
        print("\nThen run this script again.")

        # Create a demo with synthetic vibration data for stand alone use.
        print("\nGenerating demo with synthetic vibration data...")

        t = np.linspace(0.0, 60.0, 6000)

        # S curve: larger residual oscillation after slew.
        mode1_u = np.zeros_like(t)
        mode1_u[t >= 30] = (
            0.004
            * np.exp(-0.02 * (t[t >= 30] - 30))
            * np.sin(2 * np.pi * 0.4 * (t[t >= 30] - 30))
        )

        mode2_u = np.zeros_like(t)
        mode2_u[t >= 30] = (
            0.002
            * np.exp(-0.015 * (t[t >= 30] - 30))
            * np.sin(2 * np.pi * 1.3 * (t[t >= 30] - 30))
        )

        # Shaped: much smaller oscillation.
        mode1_s = mode1_u * 0.02
        mode2_s = mode2_u * 0.02

        np.savez("demo_s_curve.npz", time=t, mode1=mode1_u, mode2=mode2_u, method="s_curve")
        np.savez("demo_fourth.npz", time=t, mode1=mode1_s, mode2=mode2_s, method="fourth")

        create_blur_demo_from_modal_data(
            "demo_s_curve.npz", "demo_fourth.npz", "comet_blur_demo.png"
        )
