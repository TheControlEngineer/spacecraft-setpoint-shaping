"""
Star Field Survey Scenario - Step and Stare

Real spacecraft star surveys use "step and stare":
1. SLEW to target pointing (fast)
2. WAIT for vibrations to settle (settling time)
3. CAPTURE image when stable enough

Input shaping reduces the SETTLING TIME, not mid-slew blur.
This is because no sane mission images during active slew!

Key metrics:
- Settling time: How long after slew to wait before imaging
- Residual vibration: The oscillation amplitude after slew completes
- Throughput: More targets per orbit if settling is faster
"""

import numpy as np
from Basilisk.utilities import macros


class StepAndStareSurveyScenario:
    """Star field survey with step-and-stare imaging strategy."""
    
    def __init__(self, slew_angle=180.0, slew_duration=30.0, 
                 exposure_duration=0.2, settling_budget=10.0):
        """
        Initialize step-and-stare survey parameters.
        
        slew_angle: rotation angle in degrees (180 deg yaw)
        slew_duration: time for the slew maneuver (30s)
        exposure_duration: camera exposure time (0.2s typical)
        settling_budget: max time allowed for settling before imaging (10s)
        """
        self.slew_angle = slew_angle
        self.slew_duration = slew_duration
        self.exposure_duration = exposure_duration
        self.settling_budget = settling_budget
        
        # Step and stare timeline
        self.slew_end = slew_duration
        self.earliest_image = slew_duration  # Can't image before slew ends!
        self.latest_image = slew_duration + settling_budget
        
        # Image quality thresholds (pointing stability in microradians)
        self.threshold_strict = 1.0    # Precision astrometry
        self.threshold_normal = 10.0   # Standard star tracker
        self.threshold_relaxed = 100.0 # Coarse pointing
        
        print(f"\nStep-and-Stare Survey:")
        print(f"  Slew: {slew_angle} deg yaw in {slew_duration}s")
        print(f"  Exposure: {exposure_duration}s")
        print(f"  Settling budget: {settling_budget}s")
        print(f"  Key metric: Settling time to reach imaging threshold")
    
    def calculate_throughput(self, settling_time):
        """
        Calculate survey throughput (targets per orbit).
        
        More settling time = fewer targets per orbit.
        """
        total_time_per_target = self.slew_duration + settling_time + self.exposure_duration
        orbit_period = 90 * 60  # 90 minute orbit (seconds)
        observation_fraction = 0.5  # Half the orbit in eclipse or SAA
        
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
    """Survey camera model for star field imaging after slew settles."""
    
    def __init__(self, focal_length=2.0, pixel_size=5e-6, 
                 resolution=2048, array_length=10.0):
        """Initialize camera optics and quality thresholds."""
        self.focal_length = focal_length
        self.pixel_size = pixel_size
        self.resolution = resolution
        self.array_length = array_length
        
        # Plate scale and FOV
        self.plate_scale_rad = pixel_size / focal_length
        self.plate_scale_arcsec = np.degrees(self.plate_scale_rad) * 3600
        self.fov_rad = 2 * np.arctan(resolution * pixel_size / (2 * focal_length))
        self.fov_deg = np.degrees(self.fov_rad)
        
        # Quality thresholds for point sources
        self.threshold_sharp = 0.5       # < 0.5 px = diffraction limited
        self.threshold_acceptable = 1.0  # < 1 px = acceptable
        self.threshold_usable = 2.0      # < 2 px = usable for astrometry
        
        print(f"Survey camera: {focal_length}m f.l., {self.plate_scale_arcsec:.3f} arcsec/px, {self.fov_deg:.2f} deg FOV")
    
    def vibration_to_jitter(self, vibration_m):
        """Convert array tip displacement (m) to pointing jitter (rad)."""
        return vibration_m / self.array_length
    
    def calculate_image_quality(self, time, attitude, attitude_ref, vibration,
                                exposure_start, exposure_duration):
        """
        Calculate image quality during exposure.
        
        For star field survey, blur comes entirely from vibration-induced
        jitter around the commanded trajectory. We measure:
        
        1. Attitude tracking error (how well we follow the reference)
        2. Vibration-induced jitter (flexible mode oscillation)
        3. Total pointing stability
        
        Parameters:
        -----------
        time : array
            Time vector (seconds)
        attitude : array
            Actual spacecraft attitude (radians)
        attitude_ref : array
            Commanded/reference attitude (radians)
        vibration : array
            Solar array tip displacement (meters)
        exposure_start : float
            Start of exposure window (seconds)
        exposure_duration : float
            Duration of exposure (seconds)
        
        Returns:
        --------
        results : dict
            Image quality metrics
        """
        exposure_end = exposure_start + exposure_duration
        idx = (time >= exposure_start) & (time <= exposure_end)
        
        if np.sum(idx) == 0:
            return {
                'blur_pixels': np.inf,
                'quality': "No Data",
                'status': "ERROR"
            }
        
        # Attitude tracking error (deviation from reference trajectory)
        # This should be small for good feedforward control
        attitude_error = attitude[idx] - attitude_ref[idx]
        rms_attitude_error = np.sqrt(np.mean(attitude_error**2))
        peak_attitude_error = np.max(np.abs(attitude_error))
        
        # Vibration induced jitter. This is what input shaping suppresses
        vibration_exposure = vibration[idx]
        rms_vibration = np.sqrt(np.mean(vibration_exposure**2))
        peak_vibration = np.max(np.abs(vibration_exposure))
        
        pointing_jitter = self.vibration_to_jitter(vibration_exposure)
        rms_jitter = np.sqrt(np.mean(pointing_jitter**2))
        
        # Total pointing error
        # Blur comes from jitter (variation), not DC tracking offset
        attitude_error_variation = attitude_error - np.mean(attitude_error)
        rms_attitude_variation = np.sqrt(np.mean(attitude_error_variation**2))
        
        # Total blur = RSS of jitter and tracking variation
        total_jitter = np.sqrt(rms_attitude_variation**2 + rms_jitter**2)
        
        # Convert to image blur (pixels)
        blur_arcsec = np.degrees(total_jitter) * 3600
        blur_pixels = blur_arcsec / self.plate_scale_arcsec
        
        # Also report jitter only blur (pure input shaping metric)
        jitter_blur_arcsec = np.degrees(rms_jitter) * 3600
        jitter_blur_pixels = jitter_blur_arcsec / self.plate_scale_arcsec
        
        # Assess quality
        if blur_pixels < self.threshold_sharp:
            quality = "Sharp"
            status = "★ EXCELLENT - Diffraction limited!"
        elif blur_pixels < self.threshold_acceptable:
            quality = "Acceptable" 
            status = "GOOD - Suitable for photometry"
        elif blur_pixels < self.threshold_usable:
            quality = "Usable"
            status = "~ MARGINAL - Usable for astrometry"
        else:
            quality = "Blurred"
            status = "✗ POOR - Stars are smeared"
        
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
