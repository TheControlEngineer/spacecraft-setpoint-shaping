"""
Star Camera Simulator

Creates simple star field images and comparison videos for different
shaping methods.  The simulator generates a fixed star field and applies
Gaussian blur proportional to the spacecraft vibration rate during each
exposure window.  Higher vibration results in more star smearing.

This module prioritizes robustness and visual clarity over photometric
fidelity.  It is intended for demonstration and comparison purposes.
"""

from __future__ import annotations

import os
from typing import Dict, Any

import numpy as np

try:
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover optional dependency
    imageio = None

try:
    from scipy.ndimage import gaussian_filter
except ImportError:  # pragma: no cover optional dependency
    gaussian_filter = None


def generate_synthetic_vibration_data(method: str,
                                      duration: float = 60.0,
                                      dt: float = 0.1) -> Dict[str, Any]:
    """Generate synthetic angular rate and vibration data.

    Produces a triangular velocity slew profile followed by damped
    oscillatory vibration.  The vibration amplitude depends on the
    shaping method: 'unshaped' gives full excitation while 'fourth'
    gives near zero residual.

    Args:
        method:   shaping method name ('unshaped', 'fourth', etc.).
        duration: total simulation time in seconds.
        dt:       sample period in seconds.

    Returns:
        Dictionary with time, omega, vibration_rate, vibration_angle, and dt.
    """
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

    # Flexible vibration model: two damped sinusoids representing the
    # first and second solar array bending modes.
    f1, f2 = 0.4, 1.3
    zeta = 0.02
    omega1 = 2 * np.pi * f1
    omega2 = 2 * np.pi * f2
    omega1d = omega1 * np.sqrt(1 - zeta**2)  # Damped natural frequency, mode 1
    omega2d = omega2 * np.sqrt(1 - zeta**2)  # Damped natural frequency, mode 2

    # Excitation amplitudes depend on shaping effectiveness.
    base_amp = 0.002  # rad/s baseline
    if method == "unshaped":
        # Full excitation: both modes at their natural amplitudes
        amp1, amp2 = base_amp, base_amp * 0.6
    elif method == "fourth":
        # Near zero excitation: spectral nulling suppresses both modes
        amp1, amp2 = base_amp * 0.01, base_amp * 0.01
    else:
        # Default (e.g. s_curve): moderate excitation
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
    """Simple star field renderer for comparison videos.

    A fixed set of stars is generated once at construction using a
    deterministic seed.  Each rendered frame blurs those stars by an
    amount proportional to the RMS vibration rate during the current
    exposure window.

    Attributes:
        fov_deg:            field of view in degrees.
        resolution:         image size as (height, width) in pixels.
        exposure_time:      per frame exposure duration in seconds.
        star_density:       number of stars in the field.
        blur_amplification: gain applied to blur for visual clarity.
    """

    def __init__(self,
                 fov_deg: float = 15.0,
                 resolution: tuple[int, int] = (800, 800),
                 exposure_time: float = 0.2,
                 star_density: int = 400,
                 blur_amplification: float = 500.0) -> None:
        """Initialize the camera model and generate a fixed star field."""
        self.fov_deg = fov_deg
        self.resolution = resolution
        self.exposure_time = exposure_time
        self.star_density = int(star_density)
        self.blur_amplification = float(blur_amplification)

        # Plate scale converts angular size to pixels
        self.pixel_scale_arcsec = (self.fov_deg * 3600.0) / self.resolution[0]

        # Generate a repeatable random star field (fixed seed for consistency)
        rng = np.random.default_rng(42)
        self.star_positions = rng.random((self.star_density, 2)) * np.array(self.resolution)
        self.star_brightness = rng.uniform(0.4, 1.0, self.star_density)

    def _render_star_field(self, blur_sigma_px: float) -> np.ndarray:
        """Render a single star field frame with optional Gaussian blur."""
        h, w = self.resolution
        image = np.zeros((h, w), dtype=float)

        # Place each star as a single bright pixel
        for (x, y), brightness in zip(self.star_positions, self.star_brightness):
            xi = int(round(x))
            yi = int(round(y))
            if 0 <= xi < w and 0 <= yi < h:
                image[yi, xi] += brightness

        # Gaussian blur simulates the combined PSF broadening from
        # optical diffraction and vibration induced smear.
        if gaussian_filter is not None:
            sigma = max(0.7, blur_sigma_px)
            image = gaussian_filter(image, sigma=sigma, mode="nearest")

        # Normalize to [0, 1] for display
        if image.max() > 0:
            image = image / image.max()
        return np.clip(image, 0.0, 1.0)

    def _compute_blur_sigma(self,
                            time: np.ndarray,
                            vibration_rate: np.ndarray,
                            t_center: float) -> float:
        """Estimate blur sigma in pixels over the exposure window.

        The RMS vibration rate during the exposure is converted to an
        angular blur, then to pixel blur using the plate scale.  The
        amplification factor makes subtle jitter visible in the rendered
        frames.
        """
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
        """Generate a side by side comparison video of multiple shaping methods.

        Each frame shows the star field for every method at the same point
        in time, concatenated horizontally.  Methods with more residual
        vibration produce visibly blurrier stars.

        Args:
            simulation_data: mapping from method name to data dict.
            output_filename: file name for the output video.
            fps:             video frame rate.
        """
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
