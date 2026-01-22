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

        # Convert small-angle MRPs to arcsec in the image plane.
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
                # Point-source PSF.
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
                # Motion-blurred streaks (subsample for efficiency).
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
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Render a comet image with optional motion blur from attitude data.

        Inputs:
        - sigma_history: MRP history (N x 3). If None, render a sharp image.
        - time_history: time stamps (N,). Required if sigma_history is provided.
        - exposure_start: start time of the exposure.

        Outputs:
        - image: (H x W) float array in [0, 1].
        - stats: dictionary with RMS/peak jitter and blur in pixels.
        """
        image = np.zeros(self.resolution)

        blur_trail = None
        blur_stats = {"rms_arcsec": 0.0, "peak_arcsec": 0.0, "blur_pixels": 0.0}

        if sigma_history is not None and time_history is not None:
            jitter = self._compute_attitude_jitter(
                exposure_start, sigma_history, time_history
            )
            if len(jitter) > 1:
                blur_trail = self._angular_to_pixels(jitter)

                jitter_magnitude = np.sqrt(jitter[:, 0] ** 2 + jitter[:, 1] ** 2)
                blur_stats["rms_arcsec"] = float(np.sqrt(np.mean(jitter_magnitude**2)))
                blur_stats["peak_arcsec"] = float(np.max(jitter_magnitude))
                blur_stats["blur_pixels"] = float(
                    self._angular_to_pixels(blur_stats["peak_arcsec"])
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


def create_comet_comparison_figure(
    unshaped_data: Dict[str, np.ndarray],
    shaped_data: Dict[str, np.ndarray],
    output_filename: str = "comet_blur_comparison.png",
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Create a side-by-side comparison of unshaped vs shaped comet images.

    Inputs:
    - unshaped_data: dict containing time and sigma arrays.
    - shaped_data: dict containing time and sigma arrays.
    - output_filename: output path for the PNG figure.

    Outputs:
    - Tuple of (unshaped blur stats, shaped blur stats).
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

    img_unshaped, stats_unshaped = camera.generate_image(
        unshaped_data["sigma"], unshaped_data["time"], exposure_start
    )
    img_shaped, stats_shaped = camera.generate_image(
        shaped_data["sigma"], shaped_data["time"], exposure_start
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

    axes[1].imshow(img_unshaped, cmap=cmap, origin="lower", vmin=0, vmax=0.9)
    axes[1].set_title(
        "UNSHAPED Control\n(Bang-Bang)",
        fontsize=12,
        fontweight="bold",
        color="red",
    )
    blur_text = (
        f"Blur: {stats_unshaped['blur_pixels']:.1f} px\n"
        f"({stats_unshaped['rms_arcsec']:.1f} arcsec RMS)"
    )
    axes[1].set_xlabel(blur_text, fontsize=10, color="red")
    axes[1].axis("off")

    if stats_unshaped["blur_pixels"] > 5:
        axes[1].text(
            0.5,
            0.02,
            "UNUSABLE",
            transform=axes[1].transAxes,
            fontsize=14,
            ha="center",
            color="red",
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    axes[2].imshow(img_shaped, cmap=cmap, origin="lower", vmin=0, vmax=0.9)
    method_name = (
        shaped_data["method"].upper()
        if isinstance(shaped_data["method"], str)
        else "SHAPED"
    )
    axes[2].set_title(
        f"{method_name} Input Shaping",
        fontsize=12,
        fontweight="bold",
        color="green",
    )
    blur_text = (
        f"Blur: {stats_shaped['blur_pixels']:.1f} px\n"
        f"({stats_shaped['rms_arcsec']:.1f} arcsec RMS)"
    )
    axes[2].set_xlabel(blur_text, fontsize=10, color="green")
    axes[2].axis("off")

    if stats_shaped["blur_pixels"] < 2:
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

    if stats_unshaped["blur_pixels"] > 0.1:
        improvement = stats_unshaped["blur_pixels"] / max(
            stats_shaped["blur_pixels"], 0.1
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

    return stats_unshaped, stats_shaped


def create_blur_demo_from_modal_data(
    unshaped_file: str,
    shaped_file: str,
    output_filename: str = "comet_blur_comparison.png",
) -> None:
    """
    Load modal displacement data and render a blur comparison.

    Inputs:
    - unshaped_file: NPZ with modal displacement data for unshaped case.
    - shaped_file: NPZ with modal displacement data for shaped case.
    - output_filename: output PNG file name.
    """
    unshaped = np.load(unshaped_file, allow_pickle=True)
    shaped = np.load(shaped_file, allow_pickle=True)

    time_u = unshaped["time"]
    time_s = shaped["time"]

    # Modal displacements (default to zeros if missing).
    mode1_u = unshaped["mode1"] if "mode1" in unshaped else np.zeros_like(time_u)
    mode2_u = unshaped["mode2"] if "mode2" in unshaped else np.zeros_like(time_u)
    mode1_s = shaped["mode1"] if "mode1" in shaped else np.zeros_like(time_s)
    mode2_s = shaped["mode2"] if "mode2" in shaped else np.zeros_like(time_s)

    vibration_u = np.sqrt(mode1_u**2 + mode2_u**2)
    vibration_s = np.sqrt(mode1_s**2 + mode2_s**2)

    # Convert modal displacement to angular jitter (small-angle). Use a 4 m arm.
    arm_length = 4.0
    jitter_u_arcsec = (vibration_u / arm_length) * ARCSEC_PER_RAD
    jitter_s_arcsec = (vibration_s / arm_length) * ARCSEC_PER_RAD

    # Pixel scale for 0.5 deg FOV on 512 px.
    pixel_scale = (0.5 * 3600.0) / 512.0

    slew_end = 30.0
    post_slew_u = jitter_u_arcsec[time_u >= slew_end]
    post_slew_s = jitter_s_arcsec[time_s >= slew_end]

    rms_u = np.sqrt(np.mean(post_slew_u**2)) if len(post_slew_u) > 0 else 0.0
    rms_s = np.sqrt(np.mean(post_slew_s**2)) if len(post_slew_s) > 0 else 0.0
    peak_u = np.max(post_slew_u) if len(post_slew_u) > 0 else 0.0
    peak_s = np.max(post_slew_s) if len(post_slew_s) > 0 else 0.0

    blur_px_u = peak_u / pixel_scale
    blur_px_s = peak_s / pixel_scale

    print("\n" + "=" * 60)
    print("COMET PHOTOGRAPHY MOTION BLUR ANALYSIS")
    print("=" * 60)
    print("\nUnshaped (Bang-Bang):")
    print(f"  Angular jitter RMS:  {rms_u:.1f} arcsec")
    print(f"  Angular jitter Peak: {peak_u:.1f} arcsec")
    print(f"  Image blur:          {blur_px_u:.1f} pixels")
    print("\nFourth-Order Shaped:")
    print(f"  Angular jitter RMS:  {rms_s:.1f} arcsec")
    print(f"  Angular jitter Peak: {peak_s:.1f} arcsec")
    print(f"  Image blur:          {blur_px_s:.1f} pixels")

    if peak_s > 1e-6:
        print(f"\nImprovement: {peak_u / peak_s:.1f}x reduction in blur")

    # Create visualization.
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.2])

    # Top row: jitter time series.
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time_u, jitter_u_arcsec, "r-", alpha=0.7, linewidth=0.5)
    ax1.axvline(x=30, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Angular Jitter (arcsec)")
    ax1.set_title("Unshaped: Full Timeline", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 60])

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time_s, jitter_s_arcsec, "g-", alpha=0.7, linewidth=0.5)
    ax2.axvline(x=30, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Angular Jitter (arcsec)")
    ax2.set_title("Fourth-Order: Full Timeline", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 60])

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(
        time_u[time_u >= 30],
        jitter_u_arcsec[time_u >= 30],
        "r-",
        alpha=0.7,
        label="Unshaped",
        linewidth=0.8,
    )
    ax3.plot(
        time_s[time_s >= 30],
        jitter_s_arcsec[time_s >= 30],
        "g-",
        alpha=0.7,
        label="Fourth-order",
        linewidth=0.8,
    )
    ax3.axhline(y=pixel_scale, color="blue", linestyle=":", label="1 pixel", alpha=0.5)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Angular Jitter (arcsec)")
    ax3.set_title("Post-Slew Comparison", fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Middle row: frequency content.
    ax4 = fig.add_subplot(gs[1, :])

    from scipy import signal as sig

    dt = time_u[1] - time_u[0]
    fs = 1.0 / dt
    data_u = jitter_u_arcsec[time_u >= 30]
    data_s = jitter_s_arcsec[time_s >= 30]

    if len(data_u) > 256:
        f_u, psd_u = sig.periodogram(data_u, fs, window="hann", scaling="density")
        f_s, psd_s = sig.periodogram(data_s, fs, window="hann", scaling="density")

        ax4.semilogy(f_u, psd_u, "r-", linewidth=1.5, label="Unshaped", alpha=0.9)
        ax4.semilogy(f_s, psd_s, "g-", linewidth=1.5, label="Fourth-Order", alpha=0.9)

        modal_freqs = [0.4, 1.3]
        for freq in modal_freqs:
            ax4.axvline(x=freq, color="blue", linestyle="--", alpha=0.5, linewidth=1)

        ax4.set_xlabel("Frequency (Hz)")
        ax4.set_ylabel("Power Spectral Density (arcsec^2/Hz)")
        ax4.set_title("Jitter Power Spectrum")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim([0, 1.5])

    # Bottom row: enhanced comet images.
    def make_comet_image(blur_radius_px: float, ax, title: str, title_color: str = "black"):
        """Create a synthetic comet image with a simple blur model."""
        img_size = 384
        img = np.zeros((img_size, img_size))

        cx, cy = img_size // 2, img_size // 2
        y, x = np.ogrid[:img_size, :img_size]

        n_blur = max(1, int(blur_radius_px * 4))

        for _ in range(n_blur):
            # Random offset within the blur radius.
            if blur_radius_px > 0.5:
                dx = np.random.uniform(-blur_radius_px, blur_radius_px)
                dy = np.random.uniform(-blur_radius_px, blur_radius_px)
            else:
                dx, dy = 0.0, 0.0

            shifted_cx = cx + dx
            shifted_cy = cy + dy

            r = np.sqrt((x - shifted_cx) ** 2 + (y - shifted_cy) ** 2)
            nucleus = (1.5 / n_blur) * np.exp(-r**2 / (2 * 6**2))
            coma = (0.6 / n_blur) * np.exp(-r / 40)

            tail_angle = np.radians(45.0)
            tail_x = (x - shifted_cx) * np.cos(tail_angle) + (y - shifted_cy) * np.sin(
                tail_angle
            )
            tail_y = -(x - shifted_cx) * np.sin(tail_angle) + (y - shifted_cy) * np.cos(
                tail_angle
            )

            tail = np.zeros_like(img)
            tail_mask = tail_x < 0
            tail[tail_mask] = (
                (0.35 / n_blur)
                * np.exp(tail_x[tail_mask] / 100)
                * np.exp(
                    -tail_y[tail_mask] ** 2
                    / (2 * (20 + 0.3 * np.abs(tail_x[tail_mask])) ** 2)
                )
            )

            img += nucleus + coma + tail

        # Add background stars.
        rng = np.random.default_rng(42)
        for _ in range(20):
            sx, sy = rng.integers(30, img_size - 30, 2)
            brightness = rng.uniform(0.15, 0.35)

            if blur_radius_px > 0.5:
                for _ in range(int(blur_radius_px * 3)):
                    ddx = rng.uniform(-blur_radius_px, blur_radius_px)
                    ddy = rng.uniform(-blur_radius_px, blur_radius_px)
                    star_r = np.sqrt((x - sx - ddx) ** 2 + (y - sy - ddy) ** 2)
                    img += (brightness / (blur_radius_px * 3)) * np.exp(
                        -star_r**2 / (2 * 1.5**2)
                    )
            else:
                star_r = np.sqrt((x - sx) ** 2 + (y - sy) ** 2)
                img += brightness * np.exp(-star_r**2 / (2 * 1.5**2))

        img += rng.normal(0.0, 0.008, img.shape)
        img = np.clip(img, 0.0, 1.0)
        img = np.power(img, 0.85)

        ax.imshow(img, cmap="gray", origin="lower", vmin=0, vmax=0.85, interpolation="bilinear")
        ax.set_title(title, fontsize=12, fontweight="bold", color=title_color, pad=10)
        ax.axis("off")

        return img

    ax5 = fig.add_subplot(gs[2, 0])
    make_comet_image(0.0, ax5, "Reference\n(No Motion)")

    ax6 = fig.add_subplot(gs[2, 1])
    make_comet_image(blur_px_u, ax6, f"UNSHAPED\nBlur: {blur_px_u:.1f} px", "red")
    if blur_px_u > 3:
        ax6.text(
            0.5,
            0.05,
            "BLURRED",
            transform=ax6.transAxes,
            fontsize=12,
            ha="center",
            color="red",
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    ax7 = fig.add_subplot(gs[2, 2])
    make_comet_image(
        blur_px_s, ax7, f"FOURTH-ORDER\nBlur: {blur_px_s:.1f} px", "green"
    )
    if blur_px_s < 2:
        ax7.text(
            0.5,
            0.05,
            "SHARP",
            transform=ax7.transAxes,
            fontsize=12,
            ha="center",
            color="green",
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    fig.suptitle(
        "Comet Photography Mission: Input Shaping Impact\n"
        "(Telephoto Camera, 2-second exposure)",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(output_filename, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {output_filename}")
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("Comet Camera Simulator - Motion Blur Visualization")
    print("=" * 60)
    print("\nAligned with vizard_demo.py comet photography mission!")

    unshaped_file = None
    shaped_file = None

    possible_unshaped = [
        "vizard_demo_unshaped_filtered_pd.npz",
        "vizard_demo_unshaped_standard_pd.npz",
        "vizard_demo_unshaped.npz",
        "comparison_unshaped.npz",
    ]
    possible_shaped = [
        "vizard_demo_fourth_filtered_pd.npz",
        "vizard_demo_fourth_standard_pd.npz",
        "vizard_demo_fourth.npz",
        "comparison_fourth.npz",
    ]

    for name in possible_unshaped:
        if os.path.exists(name):
            unshaped_file = name
            break

    for name in possible_shaped:
        if os.path.exists(name):
            shaped_file = name
            break

    if unshaped_file and shaped_file:
        print("\nUsing data files:")
        print(f"  Unshaped: {unshaped_file}")
        print(f"  Shaped:   {shaped_file}")
        create_blur_demo_from_modal_data(unshaped_file, shaped_file)
    else:
        print("\nNo simulation data found. Run vizard_demo.py first:")
        print("  python vizard_demo.py unshaped --controller filtered_pd")
        print("  python vizard_demo.py fourth --controller filtered_pd")
        print("\nThen run this script again.")

        # Create a demo with synthetic vibration data for stand-alone use.
        print("\nGenerating demo with synthetic vibration data...")

        t = np.linspace(0.0, 60.0, 6000)

        # Unshaped: large oscillation after slew.
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

        np.savez("demo_unshaped.npz", time=t, mode1=mode1_u, mode2=mode2_u, method="unshaped")
        np.savez("demo_fourth.npz", time=t, mode1=mode1_s, mode2=mode2_s, method="fourth")

        create_blur_demo_from_modal_data(
            "demo_unshaped.npz", "demo_fourth.npz", "comet_blur_demo.png"
        )
