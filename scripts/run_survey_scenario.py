"""
Star Field Survey Scenario: Step and Stare

Spacecraft star surveys use step and stare operations:
  1. Slew to target pointing (fast rotation).
  2. Wait for vibrations to settle (settling time).
  3. Capture image when stable enough.

Input shaping reduces the settling time rather than mid slew blur
because no mission images during an active slew.

Key metrics:
  Settling time:       how long after the slew to wait before imaging.
  Residual vibration:  oscillation amplitude after the slew completes.
  Throughput:          number of targets per orbit (faster settling = more).
"""

import numpy as np
from Basilisk.utilities import macros


class StepAndStareSurveyScenario:
    """Star field survey with step and stare imaging strategy.

    Encapsulates the timeline parameters (slew duration, exposure time,
    settling budget) and provides methods to compute throughput and
    print the mission timeline for a single target.
    """
    
    def __init__(self, slew_angle=180.0, slew_duration=30.0, 
                 exposure_duration=0.2, settling_budget=10.0):
        """
        Initialize step and stare survey parameters.

        Args:
            slew_angle:        rotation angle in degrees (e.g. 180 deg yaw).
            slew_duration:     time for the slew manoeuvre in seconds.
            exposure_duration:  camera exposure time in seconds.
            settling_budget:   maximum time allowed for settling before imaging.
        """
        self.slew_angle = slew_angle
        self.slew_duration = slew_duration
        self.exposure_duration = exposure_duration
        self.settling_budget = settling_budget
        
        # Step and stare timeline
        self.slew_end = slew_duration
        self.earliest_image = slew_duration     # Cannot image before slew ends
        self.latest_image = slew_duration + settling_budget
        
        # Image quality thresholds (pointing stability in microradians)
        self.threshold_strict = 1.0    # Precision astrometry requirement
        self.threshold_normal = 10.0   # Standard star tracker requirement
        self.threshold_relaxed = 100.0 # Coarse pointing sufficient
        
        print(f"\nStep and Stare Survey:")
        print(f"  Slew: {slew_angle} deg yaw in {slew_duration}s")
        print(f"  Exposure: {exposure_duration}s")
        print(f"  Settling budget: {settling_budget}s")
        print(f"  Key metric: settling time to reach imaging threshold")
    
    def calculate_throughput(self, settling_time):
        """
        Calculate survey throughput in targets per orbit.

        The total time per target includes slew, settling, and exposure.
        More settling time directly reduces the number of targets that
        can be observed in one orbit.
        """
        total_time_per_target = self.slew_duration + settling_time + self.exposure_duration
        orbit_period = 90 * 60       # 90 minute LEO orbit in seconds
        observation_fraction = 0.5   # Half the orbit usable (eclipse, SAA excluded)
        
        available_time = orbit_period * observation_fraction
        targets_per_orbit = available_time / total_time_per_target
        
        return targets_per_orbit
    
    def print_mission_timeline(self, settling_time=5.0):
        """Print mission timeline for one target."""
        print(f"\nSingle Target Timeline:")
        print(f"  t=0.0s:  Start slew")
        print(f"  t={self.slew_duration:.1f}s: Slew complete, waiting...")
        print(f"  t={self.slew_duration + settling_time:.1f}s: Settled, start exposure")
        print(f"  t={self.slew_duration + settling_time + self.exposure_duration:.1f}s: Image captured")
        print(f"  Targets/orbit: {self.calculate_throughput(settling_time):.0f}")


# Keep old class name for compatibility
class StarFieldSurveyScenario(StepAndStareSurveyScenario):
    """Legacy alias for backward compatibility."""
    pass


class SurveyCamera:
    """Survey camera model for star field imaging after slew settles.

    Provides plate scale, field of view, and image quality assessment
    based on pointing jitter during the exposure window.
    """
    
    def __init__(self, focal_length=2.0, pixel_size=5e-6, 
                 resolution=2048, array_length=10.0):
        """Initialize camera optics and quality thresholds.

        Args:
            focal_length:  effective focal length in metres.
            pixel_size:    detector pixel pitch in metres.
            resolution:    detector side length in pixels.
            array_length:  solar array length used for lever arm computation.
        """
        self.focal_length = focal_length
        self.pixel_size = pixel_size
        self.resolution = resolution
        self.array_length = array_length
        
        # Plate scale and field of view derived from optics
        self.plate_scale_rad = pixel_size / focal_length
        self.plate_scale_arcsec = np.degrees(self.plate_scale_rad) * 3600
        self.fov_rad = 2 * np.arctan(resolution * pixel_size / (2 * focal_length))
        self.fov_deg = np.degrees(self.fov_rad)
        
        # Image quality thresholds for point sources (pixels of blur)
        self.threshold_sharp = 0.5       # Diffraction limited
        self.threshold_acceptable = 1.0  # Photometry quality
        self.threshold_usable = 2.0      # Astrometry quality
        
        print(f"Survey camera: {focal_length}m f.l., {self.plate_scale_arcsec:.3f} arcsec/px, {self.fov_deg:.2f} deg FOV")
    
    def vibration_to_jitter(self, vibration_m):
        """Convert array tip displacement (m) to pointing jitter (rad)."""
        return vibration_m / self.array_length
    
    def calculate_image_quality(self, time, attitude, attitude_ref, vibration,
                                exposure_start, exposure_duration):
        """
        Calculate image quality during an exposure window.

        Blur comes from two sources:
          1. Attitude tracking error (deviation from the reference trajectory).
          2. Vibration induced jitter (flexible mode oscillation).

        The DC tracking offset is removed so only the oscillatory
        component contributes to blur.

        Args:
            time:              time vector in seconds.
            attitude:          actual spacecraft attitude in radians.
            attitude_ref:      commanded reference attitude in radians.
            vibration:         solar array tip displacement in metres.
            exposure_start:    start of the exposure window in seconds.
            exposure_duration:  duration of the exposure in seconds.

        Returns:
            Dictionary of image quality metrics including blur in pixels.
        """
        exposure_end = exposure_start + exposure_duration
        idx = (time >= exposure_start) & (time <= exposure_end)
        
        if np.sum(idx) == 0:
            return {
                'blur_pixels': np.inf,
                'quality': "No Data",
                'status': "ERROR"
            }
        
        # Attitude tracking error (deviation from reference trajectory).
        # Should be small when feedforward control is effective.
        attitude_error = attitude[idx] - attitude_ref[idx]
        rms_attitude_error = np.sqrt(np.mean(attitude_error**2))
        peak_attitude_error = np.max(np.abs(attitude_error))
        
        # Vibration induced jitter: this is what input shaping suppresses.
        vibration_exposure = vibration[idx]
        rms_vibration = np.sqrt(np.mean(vibration_exposure**2))
        peak_vibration = np.max(np.abs(vibration_exposure))
        
        pointing_jitter = self.vibration_to_jitter(vibration_exposure)
        rms_jitter = np.sqrt(np.mean(pointing_jitter**2))
        
        # Remove DC offset from tracking error so only the oscillatory
        # component contributes to motion blur.
        attitude_error_variation = attitude_error - np.mean(attitude_error)
        rms_attitude_variation = np.sqrt(np.mean(attitude_error_variation**2))
        
        # Total blur is the RSS of tracking variation and jitter
        total_jitter = np.sqrt(rms_attitude_variation**2 + rms_jitter**2)
        
        # Convert angular blur to pixels via the plate scale
        blur_arcsec = np.degrees(total_jitter) * 3600
        blur_pixels = blur_arcsec / self.plate_scale_arcsec
        
        # Pure jitter blur (the input shaping metric)
        jitter_blur_arcsec = np.degrees(rms_jitter) * 3600
        jitter_blur_pixels = jitter_blur_arcsec / self.plate_scale_arcsec
        
        # Assess quality against thresholds
        if blur_pixels < self.threshold_sharp:
            quality = "Sharp"
            status = "EXCELLENT: diffraction limited"
        elif blur_pixels < self.threshold_acceptable:
            quality = "Acceptable" 
            status = "GOOD: suitable for photometry"
        elif blur_pixels < self.threshold_usable:
            quality = "Usable"
            status = "MARGINAL: usable for astrometry"
        else:
            quality = "Blurred"
            status = "POOR: stars are smeared"
        
        results = {
            'blur_pixels': blur_pixels,
            'quality': quality,
            'status': status,
            'rms_attitude_error_rad': rms_attitude_error,
            'rms_attitude_error_arcsec': np.degrees(rms_attitude_error) * 3600,
            'rms_attitude_variation_arcsec': np.degrees(rms_attitude_variation) * 3600,
            'peak_attitude_error_arcsec': np.degrees(peak_attitude_error) * 3600,
            'rms_vibration_mm': rms_vibration * 1000,
            'peak_vibration_mm': peak_vibration * 1000,
            'rms_jitter_arcsec': np.degrees(rms_jitter) * 3600,
            'jitter_blur_pixels': jitter_blur_pixels,
            'total_jitter_arcsec': np.degrees(total_jitter) * 3600,
            'exposure_start': exposure_start,
            'exposure_end': exposure_end,
            'exposure_duration': exposure_duration
        }
        
        return results
    
    def print_image_quality_report(self, results, method_name="Unknown"):
        """Print image quality metrics."""
        print(f"\n{method_name.upper()} Image Quality:")
        print(f"  Blur: {results['blur_pixels']:.3f} px ({results['quality']})")
        print(f"  Jitter: {results['rms_jitter_arcsec']:.3f} arcsec RMS")
        print(f"  Vibration: {results['rms_vibration_mm']:.4f} mm RMS")


if __name__ == "__main__":
    # Test the scenario
    scenario = StarFieldSurveyScenario(
        slew_angle=180.0,
        slew_duration=30.0,
        exposure_duration=1.0  # 1 second exposures
    )
    
    scenario.print_mission_timeline()
    
    # Test camera
    camera = SurveyCamera()
