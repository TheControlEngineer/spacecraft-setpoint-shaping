"""
Star Camera Simulator

Creates simple star field images and comparison videos for different
shaping methods. The implementation prioritizes robustness over fidelity.
"""

from __future__ import annotations

import os
from typing import Dict, Any

import numpy as np

try:
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover - optional dependency
    imageio = None

try:
    from scipy.ndimage import gaussian_filter
except ImportError:  # pragma: no cover - optional dependency
    gaussian_filter = None


def generate_synthetic_vibration_data(method: str,
                                      duration: float = 60.0,
                                      dt: float = 0.1) -> Dict[str, Any]:
    """Generate synthetic angular rate and vibration data."""
    time = np.arange(0, duration + dt, dt)

    # Simple slew profile (triangular angular rate)
    slew_time = 30.0
    omega_peak = 0.02  # rad/s
    omega_z = np.zeros_like(time)
    for i, t in enumerate(time):
        if t <= slew_time / 2:
            omega_z[i] = omega_peak * (t / (slew_time / 2))
        elif t <= slew_time:
            omega_z[i] = omega_peak * (1 - (t - slew_time / 2) / (slew_time / 2))
        else:
            omega_z[i] = 0.0

    # Flexible vibration model
    f1, f2 = 0.4, 1.3
    zeta = 0.02
    omega1 = 2 * np.pi * f1
    omega2 = 2 * np.pi * f2
    omega1d = omega1 * np.sqrt(1 - zeta**2)
    omega2d = omega2 * np.sqrt(1 - zeta**2)

    base_amp = 0.002  # rad/s
    if method == "unshaped":
        amp1, amp2 = base_amp, base_amp * 0.6
    elif method == "fourth":
        amp1, amp2 = base_amp * 0.01, base_amp * 0.01
    else:
        amp1, amp2 = base_amp, base_amp * 0.6

    t = time - time[0]
    vibration_rate = (
        amp1 * np.exp(-zeta * omega1 * t) * np.sin(omega1d * t) +
        amp2 * np.exp(-zeta * omega2 * t) * np.sin(omega2d * t)
    )

    vibration_angle = np.cumsum(vibration_rate) * dt

    return {
        "time": time,
        "omega_x": np.zeros_like(time),
        "omega_y": np.zeros_like(time),
        "omega_z": omega_z,
        "vibration_rate": vibration_rate,
        "vibration_angle": vibration_angle,
        "dt": dt,
    }


class StarCameraSimulator:
    """Simple star field renderer for comparison videos."""

    def __init__(self,
                 fov_deg: float = 15.0,
                 resolution: tuple[int, int] = (800, 800),
                 exposure_time: float = 0.2,
                 star_density: int = 400,
                 blur_amplification: float = 500.0) -> None:
        self.fov_deg = fov_deg
        self.resolution = resolution
        self.exposure_time = exposure_time
        self.star_density = int(star_density)
        self.blur_amplification = float(blur_amplification)

        self.pixel_scale_arcsec = (self.fov_deg * 3600.0) / self.resolution[0]

        rng = np.random.default_rng(42)
        self.star_positions = rng.random((self.star_density, 2)) * np.array(self.resolution)
        self.star_brightness = rng.uniform(0.4, 1.0, self.star_density)

    def _render_star_field(self, blur_sigma_px: float) -> np.ndarray:
        """Render a single star field frame."""
        h, w = self.resolution
        image = np.zeros((h, w), dtype=float)

        for (x, y), brightness in zip(self.star_positions, self.star_brightness):
            xi = int(round(x))
            yi = int(round(y))
            if 0 <= xi < w and 0 <= yi < h:
                image[yi, xi] += brightness

        # Apply blur if available
        if gaussian_filter is not None:
            sigma = max(0.7, blur_sigma_px)
            image = gaussian_filter(image, sigma=sigma, mode="nearest")

        # Normalize and clip
        if image.max() > 0:
            image = image / image.max()
        return np.clip(image, 0.0, 1.0)

    def _compute_blur_sigma(self,
                            time: np.ndarray,
                            vibration_rate: np.ndarray,
                            t_center: float) -> float:
        """Estimate blur sigma in pixels over the exposure window."""
        t_start = t_center
        t_end = t_center + self.exposure_time
        mask = (time >= t_start) & (time <= t_end)
        if not np.any(mask):
            return 0.0

        vib = vibration_rate[mask]
        rms_rate = float(np.sqrt(np.mean(vib**2)))
        blur_rad = rms_rate * self.exposure_time
        blur_arcsec = np.degrees(blur_rad) * 3600.0
        blur_px = blur_arcsec / self.pixel_scale_arcsec

        # Scale for visualization
        blur_px *= self.blur_amplification
        return float(min(50.0, max(0.0, blur_px)))

    def generate_comparison_video(self,
                                  simulation_data: Dict[str, Dict[str, Any]],
                                  output_filename: str = "star_camera_comparison.mp4",
                                  fps: int = 10) -> None:
        """Generate a simple side-by-side comparison video."""
        methods = list(simulation_data.keys())
        if not methods:
            print("No simulation data provided.")
            return

        # Determine frame times based on shortest duration
        max_time = min(simulation_data[m]["time"][-1] for m in methods)
        frame_times = np.arange(0, max_time, 1.0 / fps)

        frames = []
        for t_center in frame_times:
            panels = []
            for method in methods:
                data = simulation_data[method]
                blur_sigma = self._compute_blur_sigma(
                    np.asarray(data["time"]),
                    np.asarray(data["vibration_rate"]),
                    t_center,
                )
                panels.append(self._render_star_field(blur_sigma))

            # Concatenate panels with a small separator
            separator = np.ones((self.resolution[0], 8)) * 0.1
            frame = panels[0]
            for panel in panels[1:]:
                frame = np.hstack([frame, separator, panel])
            frames.append((frame * 255).astype(np.uint8))

        if imageio is None:
            frames_dir = os.path.splitext(output_filename)[0] + "_frames"
            os.makedirs(frames_dir, exist_ok=True)
            for i, frame in enumerate(frames):
                try:
                    import matplotlib.pyplot as plt
                except ImportError:
                    break
                plt.imsave(os.path.join(frames_dir, f"frame_{i:04d}.png"), frame, cmap="gray")
            print(f"imageio not available; saved frames to {frames_dir}")
            return

        with imageio.get_writer(output_filename, fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)

        print(f"Saved: {output_filename}")
