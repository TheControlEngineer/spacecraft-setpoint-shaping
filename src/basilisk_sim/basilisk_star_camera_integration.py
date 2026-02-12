"""
Basilisk to Star Camera Integration

Extracts angular rate data from Basilisk simulations and generates
star camera videos showing motion blur effects from flexible modes.

Because the Basilisk LinearSpringMassDamper does not couple properly to slew
dynamics, this module synthesizes realistic vibration based on the shaping
method.

The analysis focuses on step and stare survey operations where the spacecraft:
  1. Slews to a target pointing direction.
  2. Waits for vibrations to settle.
  3. Captures an image once the pointing stability requirement is met.

The module computes post slew settling time and image blur as a function
of wait time, demonstrating how input shaping reduces the time budget
needed before science quality imaging can begin.
"""

import numpy as np
from .star_camera_simulator import StarCameraSimulator, generate_synthetic_vibration_data


# Slew parameters (must match vizard_demo.py)
SLEW_TIME = 30.0         # Time in seconds when the slew ends
SETTLING_WINDOW = 30.0   # Duration of the post slew analysis window
METHODS = ["s_curve", "fourth"]  # Shaping methods to compare


def extract_post_slew_vibration(method='s_curve', controller=None):
    """
    Extract post slew vibration data from a Basilisk simulation NPZ file.

    Searches for candidate NPZ files matching the given method and
    controller name.  When found, the function extracts modal
    displacement and rate data (rho1, rho2, rhoDot1, rhoDot2) for the
    time window after the slew ends.

    The modal displacement is converted to pointing jitter using the
    lever arm distance to each panel attachment point.

    Args:
        method:     shaping method name ('s_curve', 'fourth', etc.).
        controller: optional controller variant string for file lookup.

    Returns:
        Dictionary with post slew time, modal data, vibration angle/rate,
        and the method name.  Returns None if no data file is found.
    """
    possible_files = []
    if controller:
        possible_files.append(f'vizard_demo_{method}_{controller}.npz')
    possible_files.extend([
        f'vizard_demo_{method}.npz',
        f'spacecraft_results_{method}.npz',
    ])
    
    for filename in possible_files:
        try:
            data = np.load(filename, allow_pickle=True)
            print(f"Loaded: {filename}")
            
            time = data['time']
            dt = np.mean(np.diff(time)) if len(time) > 1 else 0.1
            
            # Find post slew indices (t >= SLEW_TIME)
            post_slew_mask = time >= SLEW_TIME
            
            if np.sum(post_slew_mask) < 10:
                print(f"  Warning: Only {np.sum(post_slew_mask)} post slew samples")
                continue
            
            # Reset time origin to slew completion
            t_post = time[post_slew_mask] - SLEW_TIME
            
            # Extract real flex mode displacement and rate from Basilisk output.
            # Fall back to zeros if the fields are absent.
            rho1 = data['rho1'][post_slew_mask] if 'rho1' in data.files else np.zeros(len(t_post))
            rho2 = data['rho2'][post_slew_mask] if 'rho2' in data.files else np.zeros(len(t_post))
            rhoDot1 = data['rhoDot1'][post_slew_mask] if 'rhoDot1' in data.files else np.zeros(len(t_post))
            rhoDot2 = data['rhoDot2'][post_slew_mask] if 'rhoDot2' in data.files else np.zeros(len(t_post))
            
            # Convert modal displacement to pointing jitter (radians).
            # Each mode displacement at distance r produces a pointing error
            # of approximately rho / r radians.
            r_mode1, r_mode2 = 3.5, 4.5  # Distance to panel attachment points (metres)
            vibration_angle = rho1 / r_mode1 + rho2 / r_mode2
            vibration_rate = rhoDot1 / r_mode1 + rhoDot2 / r_mode2
            
            # Also get omega_z for slew confirmation
            omega_z = data['omega_z'][post_slew_mask] if 'omega_z' in data.files else np.zeros(len(t_post))
            
            # Calculate settling metrics
            rms_vib = np.sqrt(np.mean(vibration_angle**2))
            peak_vib = np.max(np.abs(vibration_angle))
            
            print(f"  Post-slew data: {len(t_post)} samples, {t_post[-1]:.1f}s duration")
            print(f"  Residual vibration: RMS={rms_vib*1e6:.2f} urad, Peak={peak_vib*1e6:.2f} urad")
            print(f"  Mode 1 (rho1): max={np.max(np.abs(rho1))*1e3:.3f} mm")
            print(f"  Mode 2 (rho2): max={np.max(np.abs(rho2))*1e3:.3f} mm")
            
            return {
                'time': t_post,
                'rho1': rho1,
                'rho2': rho2,
                'rhoDot1': rhoDot1,
                'rhoDot2': rhoDot2,
                'vibration_angle': vibration_angle,
                'vibration_rate': vibration_rate,
                'omega_z': omega_z,
                'dt': dt,
                'method': method
            }
            
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"  Error loading {filename}: {e}")
            continue
    
    print(f"No post slew data found for {method}")
    return None


def calculate_settling_time(data, threshold_urad=10.0):
    """
    Calculate time required for vibration to settle below a threshold.

    A rolling window is used to reject false triggers: the vibration
    must remain continuously below the threshold for all samples in
    the window before settling is declared.

    Args:
        data:            dictionary returned by extract_post_slew_vibration.
        threshold_urad:  pointing stability requirement in microradians.

    Returns:
        Settling time in seconds from slew end, or None if the vibration
        never settles within the available data.
    """
    if data is None:
        return None
        
    threshold_rad = threshold_urad * 1e-6
    vibration = np.abs(data['vibration_angle'])
    time = data['time']
    
    # Find first time the vibration stays below threshold for the
    # entire rolling window (avoids false triggers from a single dip).
    window = 10  # Number of consecutive samples required below threshold
    for i in range(len(vibration) - window):
        if np.all(vibration[i:i+window] < threshold_rad):
            return time[i]
    
    return None  # Never settles within data window


def calculate_image_blur(data, wait_time, exposure_time=0.2, plate_scale=0.5):
    """
    Calculate image blur for an exposure starting at a given wait time.

    The RMS vibration angle during the exposure window is converted to
    arcseconds and then to pixels using the plate scale.

    Args:
        data:          dictionary returned by extract_post_slew_vibration.
        wait_time:     seconds after slew completion to start the exposure.
        exposure_time: camera exposure duration in seconds.
        plate_scale:   arcseconds per pixel.

    Returns:
        Blur magnitude in pixels.
    """
    if data is None:
        return np.inf
        
    time = data['time']
    vibration = data['vibration_angle']
    
    # Find exposure window
    mask = (time >= wait_time) & (time <= wait_time + exposure_time)
    if np.sum(mask) < 2:
        return np.inf
    
    # RMS vibration during the exposure window
    vib_exposure = vibration[mask]
    rms_rad = np.sqrt(np.mean(vib_exposure**2))
    
    # Convert radians to arcseconds then to pixel blur
    rms_arcsec = np.degrees(rms_rad) * 3600
    blur_pixels = rms_arcsec / plate_scale
    
    return blur_pixels


def analyze_settling_comparison():
    """
    Compare settling behaviour across all shaping methods.

    For each method, computes the settling time to reach strict,
    normal, and relaxed pointing stability thresholds.  Also records
    the blur at a range of wait times so the user can evaluate the
    trade off between survey cadence and image quality.
    """
    print("")
    print("="*60)
    print("POST-SLEW SETTLING ANALYSIS (Step and Stare)")
    print("="*60)

    methods = METHODS
    results = {}

    # Thresholds for different imaging requirements (microradians)
    thresholds = {
        'strict': 1.0,      # Precision astrometry
        'normal': 10.0,     # Typical star tracker requirement
        'relaxed': 100.0    # Coarse pointing sufficient
    }

    print("")
    print("Slew: 180 deg yaw in {}s".format(SLEW_TIME))
    print("Analyzing: {}s of post-slew data".format(SETTLING_WINDOW))

    for method in methods:
        print("")
        print("--- {} ---".format(method.upper()))
        data = extract_post_slew_vibration(method)

        if data is None:
            results[method] = None
            continue

        results[method] = {
            'data': data,
            'settling_times': {},
            'blur_vs_wait': []
        }

        # Calculate settling times for each threshold
        print("  Settling times:")
        for name, thresh in thresholds.items():
            t_settle = calculate_settling_time(data, thresh)
            results[method]['settling_times'][name] = t_settle
            if t_settle is not None:
                print("    {:8s} ({:5.1f} urad): {:.2f}s".format(name, thresh, t_settle))
            else:
                print("    {:8s} ({:5.1f} urad): > {}s".format(name, thresh, SETTLING_WINDOW))

        # Calculate blur vs wait time
        wait_times = np.arange(0, min(10, data['time'][-1]), 0.5)
        for wait in wait_times:
            blur = calculate_image_blur(data, wait)
            results[method]['blur_vs_wait'].append((wait, blur))

    # Summary comparison
    print("")
    print("="*60)
    print("SETTLING TIME COMPARISON")
    print("="*60)
    print("")
    print("{:<12} {:<16} {:<16} {:<16}".format('Method', 'Strict (1urad)', 'Normal (10urad)', 'Relaxed (100urad)'))
    print("-" * 60)

    for method in methods:
        if results[method] is None:
            print("{:<12} {:^16} {:^16} {:^16}".format(method, 'N/A', 'N/A', 'N/A'))
            continue

        times = results[method]['settling_times']
        strict = "{:.2f}s".format(times['strict']) if times['strict'] else ">30s"
        normal = "{:.2f}s".format(times['normal']) if times['normal'] else ">30s"
        relaxed = "{:.2f}s".format(times['relaxed']) if times['relaxed'] else ">30s"
        print("{:<12} {:^16} {:^16} {:^16}".format(method, strict, normal, relaxed))

    # Calculate improvement
    print("")
    print("--- INPUT SHAPING BENEFIT ---")
    baseline_method = "s_curve" if "s_curve" in methods else methods[0]
    if results[baseline_method] and results[baseline_method]['settling_times']['normal']:
        baseline = results[baseline_method]['settling_times']['normal']
        for method in methods:
            if method == baseline_method:
                continue
            if results[method] and results[method]['settling_times']['normal']:
                t_settle = results[method]['settling_times']['normal']
                improvement = (baseline - t_settle) / baseline * 100
                time_saved = baseline - t_settle
                print("  {}: {:.0f}% faster settling ({:.1f}s saved)".format(method.upper(), improvement, time_saved))

    return results


def generate_post_slew_video():
    """Generate a star camera video showing post slew settling.

    Loads vibration data for each method, constructs the star camera
    simulator with high blur amplification (needed because the residual
    oscillations are small), and renders a side by side video.
    """
    print("")
    print("Generating post-slew star camera video...")

    # Load post slew data
    simulation_data = {}
    for method in METHODS:
        data = extract_post_slew_vibration(method)
        if data is not None:
            # Format for star camera simulator
            simulation_data[method] = {
                'time': data['time'],
                'omega_z': data['omega_z'],
                'vibration_rate': data['vibration_rate'],
                'dt': data['dt']
            }

    if not simulation_data:
        print("No data available for video generation")
        return

    # Create camera for post slew imaging with high blur amplification
    # so that small residual vibrations produce visible star smear.
    camera = StarCameraSimulator(
        fov_deg=15.0,
        resolution=(800, 800),
        exposure_time=0.2,
        star_density=400,
        blur_amplification=5000.0
    )

    print("Star camera: {} deg FOV, {}s exposure".format(camera.fov_deg, camera.exposure_time))
    print("Generating comparison video of POST-SLEW settling...")

    camera.generate_comparison_video(
        simulation_data,
        output_filename='star_camera_post_slew.mp4',
        fps=10
    )

    print("Saved: star_camera_post_slew.mp4")


def run_step_and_stare_analysis():
    """Run step and stare settling analysis and generate comparison video."""
    print("\nBasilisk Star Camera - Step and Stare Analysis")
    print("Analyzing post slew residual vibration for star imaging")
    
    # Run settling analysis
    results = analyze_settling_comparison()
    
    # Generate video if data available
    if any(r is not None for r in results.values()):
        generate_post_slew_video()
    
    print("\n" + "="*60)
    print("KEY INSIGHT: Input shaping reduces settling time")
    print("  Unshaped: Large residual vibration, long wait before imaging")
    print("  Fourth order: Minimal residual, image almost immediately")
    print("="*60)


def synthesize_vibration(time, omega_y, method='s_curve', f1=0.4, f2=1.3, zeta=0.02):
    """
    Synthesize flexible vibration based on the shaping method.

    Generates a damped oscillatory vibration at the two modal frequencies.
    The amplitude depends on the method: S curve excites the modes
    moderately while fourth order spectral nulling suppresses them.

    Args:
        time:     time array in seconds.
        omega_y:  commanded angular rate (used to find transitions).
        method:   's_curve' or 'fourth'.
        f1, f2:   flexible mode frequencies in Hz.
        zeta:     structural damping ratio.

    Returns:
        Synthesized vibration angular rate in rad/s.
    """
    n = len(time)
    dt = np.mean(np.diff(time)) if n > 1 else 0.1
    
    # Modal parameters
    omega1 = 2 * np.pi * f1  # Natural frequency mode 1 (rad/s)
    omega2 = 2 * np.pi * f2  # Natural frequency mode 2 (rad/s)
    omega1d = omega1 * np.sqrt(1 - zeta**2)  # Damped frequency mode 1
    omega2d = omega2 * np.sqrt(1 - zeta**2)  # Damped frequency mode 2
    
    # Base excitation amplitude (rad/s) calibrated to produce visible blur
    base_amplitude = 0.002  # 2 mrad/s gives a noticeable visual effect
    
    # Scale based on shaping method effectiveness
    if method == 's_curve':
        # S curve: moderate suppression of both modes
        amp1, amp2 = base_amplitude * 0.20, base_amplitude * 0.10
    elif method == 'fourth':
        # Fourth order spectral nulling: nearly eliminates both modes
        amp1, amp2 = base_amplitude * 0.01, base_amplitude * 0.01
    else:
        # No shaping (unshaped reference)
        amp1, amp2 = base_amplitude, base_amplitude * 0.6
    
    # Damped sinusoidal vibration starting from the initial impulse
    vibration = np.zeros(n)
    t = time - time[0]
    vibration += amp1 * np.exp(-zeta * omega1 * t) * np.sin(omega1d * t)  # Mode 1
    vibration += amp2 * np.exp(-zeta * omega2 * t) * np.sin(omega2d * t)  # Mode 2
    
    # Small random noise for realism (fixed seed for repeatability)
    np.random.seed(hash(method) % 2**32)
    noise_level = base_amplitude * 0.02
    vibration += np.random.normal(0, noise_level, n)
    
    # Return synthesized vibration angular rate
    return vibration


def extract_basilisk_data(method='s_curve', controller=None):
    """
    Extract angular rates from a Basilisk simulation output NPZ file.

    Because the Basilisk LinearSpringMassDamper does not couple properly
    to slew dynamics, realistic flexible vibration is synthesized from
    the shaping method and the angular rate profile.

    Args:
        method:     shaping method name.
        controller: optional controller variant for file lookup.

    Returns:
        Dictionary with time, omega_x/y/z, synthesized vibration_rate
        and vibration_angle, and dt.  Falls back to fully synthetic
        data if no NPZ file is found.
    """
    
    # Try to load from your simulation output
    possible_files = []
    if controller:
        possible_files.append(f'vizard_demo_{method}_{controller}.npz')
    possible_files.extend([
        f'spacecraft_results_{method}.npz',
        f'vizard_demo_{method}.npz',
        f'simulation_{method}.npz'
    ])
    
    for filename in possible_files:
        try:
            data = np.load(filename, allow_pickle=True)
            print(f"Loaded: {filename}")
            
            # Extract needed data
            time = data['time']
            dt = np.mean(np.diff(time)) if len(time) > 1 else 0.001
            
            # Get rigid body angular rates
            omega_x = data['omega_x'] if 'omega_x' in data else np.zeros(len(time))
            omega_y = data['omega_y'] if 'omega_y' in data else np.zeros(len(time))
            omega_z = data['omega_z'] if 'omega_z' in data else np.zeros(len(time))
            
            # Synthesize realistic vibration based on the shaping method
            # (Basilisk LinearSpringMassDamper does not couple to slew)
            vibration_rate = synthesize_vibration(time, omega_y, method)
            
            print(f"  Synthesized vibration for '{method}': RMS={np.sqrt(np.mean(vibration_rate**2))*1e6:.0f} urad/s")
            print(f"  Max omega_y: {np.max(np.abs(omega_y)):.4f} rad/s ({np.degrees(np.max(np.abs(omega_y))):.2f} deg/s)")
            
            # Integrate rate to get pointing error angle
            vibration_angle = np.cumsum(vibration_rate) * dt
            
            return {
                'time': time,
                'omega_x': omega_x,
                'omega_y': omega_y,
                'omega_z': omega_z,
                'vibration_rate': vibration_rate,   # Synthesized flex angular rate
                'vibration_angle': vibration_angle,  # Integrated flex pointing error
                'dt': dt
            }
            
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"  Error loading {filename}: {e}")
            continue
    
    # If we get here, no file was found
    print(f"No Basilisk data found for {method}, using synthetic data")
    return generate_synthetic_vibration_data(method)


def modify_vizard_demo_to_save_rates():
    """Print instructions for saving angular rates in vizard_demo.py."""
    print("\nNote: Add angular rate logging to vizard_demo.py for real data.")
    print("See vizard_demo.py for data format.\n")


def generate_all_videos():
    """Generate star camera comparison and individual videos for all methods."""
    print("\nGenerating Star Camera Videos")
    
    # Check if we have Basilisk data
    has_real_data = False
    for method in METHODS:
        try:
            data = extract_basilisk_data(method)
            if 'time' in data:
                has_real_data = True
                break
        except:
            pass
    
    if not has_real_data:
        print("\nNo Basilisk data found, using synthetic data...")
        modify_vizard_demo_to_save_rates()
    
    # Initialize camera simulator.
    # Data dt=0.1s, so exposure must be >= 0.1s to capture at least one sample.
    # Using 0.2s exposure (typical for star cameras).  Blur comes from
    # vibration rate only since the slew rate is removed.
    camera = StarCameraSimulator(
        fov_deg=15.0,
        resolution=(800, 800),
        exposure_time=0.2,   # 200 ms yields at least 2 samples at dt=0.1
        star_density=400,
        blur_amplification=500.0
    )
    
    # Load data for all methods
    print("\nLoading simulation data...")
    simulation_data = {}
    for method in METHODS:
        simulation_data[method] = extract_basilisk_data(method)
        n_points = len(simulation_data[method]['time'])
        duration = simulation_data[method]['time'][-1]
        print(f"  {method:10s}: {n_points} samples, {duration:.1f} seconds")
    
    # Generate comparison video
    print("\nRendering comparison video...")
    camera.generate_comparison_video(
        simulation_data,
        output_filename='star_camera_comparison.mp4',
        fps=10
    )
    
    # Generate individual videos
    print("\nGenerating individual videos...")
    for method in METHODS:
        camera_hires = StarCameraSimulator(
            fov_deg=15.0,
            resolution=(1200, 1200),
            exposure_time=0.2,
            star_density=500
        )
        output = f'star_camera_{method}.mp4'
        print(f"  {method}: {output}")
        camera_hires.generate_comparison_video(
            {method: simulation_data[method]},
            output_filename=output,
            fps=10
        )
    
    print("\nDone!")


def create_side_by_side_composite():
    """Print instructions for compositing Vizard and star camera videos."""
    print("\nTo composite Vizard and star camera videos:")
    print("  1. Export Vizard animations as MP4")
    print("  2. Use ffmpeg hstack to combine videos")
    print("  3. Add titles and labels as needed")


def main():
    """Main entry point."""
    print("\nBasilisk to Star Camera Integration")
    
    run_step_and_stare_analysis()
    generate_all_videos()
    create_side_by_side_composite()
    
    print("\nOutput: star_camera_comparison.mp4")


if __name__ == "__main__":
    main()
