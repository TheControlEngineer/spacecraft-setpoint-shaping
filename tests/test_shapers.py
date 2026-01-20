"""
Unit tests for input shaper implementations.

Tests verify:
- Correct number of impulses
- Unity gain constraint (sum of amplitudes = 1)
- Timing relationships
- Amplitude values for known cases
- Comparison with published literature values
"""

import numpy as np
import pytest
import input_shaping.shapers as shaper_module
from input_shaping import (
    ZV,
    ZVD,
    ZVDD,
    EI,
    design_shaper,
    convolve_shapers,
    design_multimode_cascaded,
    design_multimode_simultaneous,
)


class TestZVShaper:
    """Tests for Zero Vibration (ZV) shaper"""
    
    def test_zv_returns_two_impulses(self):
        """ZV should always return exactly 2 impulses"""
        A, t, K = ZV(omega_n=np.pi, zeta=0.02)
        assert len(A) == 2, "ZV should have 2 impulses"
        assert len(t) == 2, "ZV should have 2 time points"
    
    def test_zv_unity_gain(self):
        """Sum of amplitudes should equal 1.0 (unity gain)"""
        A, t, K = ZV(omega_n=np.pi, zeta=0.02)
        assert np.isclose(np.sum(A), 1.0), f"Sum of amplitudes = {np.sum(A)}, expected 1.0"
    
    def test_zv_first_impulse_at_zero(self):
        """First impulse should be at t=0"""
        A, t, K = ZV(omega_n=np.pi, zeta=0.02)
        assert t[0] == 0.0, "First impulse must be at t=0"
    
    def test_zv_timing_half_period(self):
        """Second impulse should be at half the damped period"""
        omega_n = 2 * np.pi  # 1 Hz
        zeta = 0.02
        A, t, K = ZV(omega_n, zeta)
        
        omega_d = omega_n * np.sqrt(1 - zeta**2)
        expected_t2 = np.pi / omega_d
        
        assert np.isclose(t[1], expected_t2), \
            f"Second impulse at {t[1]}, expected {expected_t2}"
    
    def test_zv_undamped_case(self):
        """For undamped system (ζ=0), amplitudes should be exactly [0.5, 0.5]"""
        A, t, K = ZV(omega_n=np.pi, zeta=0.0)
        assert np.allclose(A, [0.5, 0.5]), \
            f"Undamped ZV should be [0.5, 0.5], got {A}"
    
    def test_zv_first_amplitude_larger_with_damping(self):
        """With damping, first amplitude should be larger than second"""
        A, t, K = ZV(omega_n=np.pi, zeta=0.02)
        assert A[0] > A[1], \
            f"With damping, A1 ({A[0]}) should be > A2 ({A[1]})"
    
    def test_zv_different_frequencies(self):
        """ZV should work for various frequencies"""
        frequencies = [0.1, 0.5, 1.0, 5.0, 10.0]  # Hz
        zeta = 0.02
        
        for f in frequencies:
            omega_n = 2 * np.pi * f
            A, t, K = ZV(omega_n, zeta)
            
            assert len(A) == 2
            assert np.isclose(np.sum(A), 1.0)
            assert t[0] == 0.0
            assert t[1] > 0.0


class TestZVDShaper:
    """Tests for Zero Vibration Derivative (ZVD) shaper"""
    
    def test_zvd_returns_three_impulses(self):
        """ZVD should always return exactly 3 impulses"""
        A, t, K = ZVD(omega_n=np.pi, zeta=0.02)
        assert len(A) == 3, "ZVD should have 3 impulses"
        assert len(t) == 3, "ZVD should have 3 time points"
    
    def test_zvd_unity_gain(self):
        """Sum of amplitudes should equal 1.0"""
        A, t, K = ZVD(omega_n=np.pi, zeta=0.02)
        assert np.isclose(np.sum(A), 1.0), f"Sum = {np.sum(A)}, expected 1.0"
    
    def test_zvd_equal_spacing(self):
        """Impulses should be equally spaced by half-periods"""
        omega_n = 2 * np.pi
        zeta = 0.02
        A, t, K = ZVD(omega_n, zeta)
        
        spacing1 = t[1] - t[0]
        spacing2 = t[2] - t[1]
        
        assert np.isclose(spacing1, spacing2), \
            f"Spacings not equal: {spacing1} vs {spacing2}"
    
    def test_zvd_undamped_amplitudes(self):
        """For undamped case, amplitudes should be [1/4, 1/2, 1/4]"""
        A, t, K = ZVD(omega_n=np.pi, zeta=0.0)
        expected = np.array([0.25, 0.5, 0.25])
        assert np.allclose(A, expected), \
            f"Undamped ZVD should be {expected}, got {A}"
    
    def test_zvd_middle_amplitude_largest(self):
        """Middle amplitude should be largest"""
        A, t, K = ZVD(omega_n=np.pi, zeta=0.02)
        assert A[1] > A[0] and A[1] > A[2], \
            "Middle amplitude should be largest"
    
    def test_zvd_twice_zv_duration(self):
        """ZVD duration should be twice ZV duration"""
        omega_n = np.pi
        zeta = 0.02
        
        A_zv, t_zv, K = ZV(omega_n, zeta)
        A_zvd, t_zvd, K = ZVD(omega_n, zeta)
        
        assert np.isclose(t_zvd[-1], 2 * t_zv[-1]), \
            f"ZVD duration {t_zvd[-1]} should be 2x ZV duration {t_zv[-1]}"


class TestZVDDShaper:
    """Tests for Zero Vibration Double Derivative (ZVDD) shaper"""
    
    def test_zvdd_returns_four_impulses(self):
        """ZVDD should always return exactly 4 impulses"""
        A, t = ZVDD(omega_n=np.pi, zeta=0.02)
        assert len(A) == 4, "ZVDD should have 4 impulses"
        assert len(t) == 4, "ZVDD should have 4 time points"
    
    def test_zvdd_unity_gain(self):
        """Sum of amplitudes should equal 1.0"""
        A, t = ZVDD(omega_n=np.pi, zeta=0.02)
        assert np.isclose(np.sum(A), 1.0), f"Sum = {np.sum(A)}, expected 1.0"
    
    def test_zvdd_equal_spacing(self):
        """All impulses equally spaced by half-periods"""
        omega_n = 2 * np.pi
        zeta = 0.02
        A, t = ZVDD(omega_n, zeta)
        
        spacings = [t[i+1] - t[i] for i in range(3)]
        
        for i in range(len(spacings)-1):
            assert np.isclose(spacings[i], spacings[i+1]), \
                f"Unequal spacings: {spacings}"
    
    def test_zvdd_undamped_amplitudes(self):
        """For undamped case, amplitudes should be [1/8, 3/8, 3/8, 1/8]"""
        A, t = ZVDD(omega_n=np.pi, zeta=0.0)
        expected = np.array([0.125, 0.375, 0.375, 0.125])
        assert np.allclose(A, expected, atol=1e-6), \
            f"Undamped ZVDD should be {expected}, got {A}"
    
    def test_zvdd_binomial_pattern(self):
        """Undamped amplitudes should follow binomial pattern [1,3,3,1]/8"""
        A, t = ZVDD(omega_n=np.pi, zeta=0.0)
        
        # Check ratio A2/A1 ≈ 3
        assert np.isclose(A[1]/A[0], 3.0), "A2/A1 should be ≈3"
        # Check symmetry
        assert np.isclose(A[0], A[3]), "A1 and A4 should be equal"
        assert np.isclose(A[1], A[2]), "A2 and A3 should be equal"
    
    def test_zvdd_three_times_zv_duration(self):
        """ZVDD duration should be three times ZV duration"""
        omega_n = np.pi
        zeta = 0.02
        
        A_zv, t_zv, K = ZV(omega_n, zeta)
        A_zvdd, t_zvdd = ZVDD(omega_n, zeta)
        
        assert np.isclose(t_zvdd[-1], 3 * t_zv[-1]), \
            f"ZVDD duration {t_zvdd[-1]} should be 3x ZV duration {t_zv[-1]}"


class TestEIShaper:
    """Tests for Extra-Insensitive (EI) shaper"""
    
    def test_ei_returns_three_impulses(self):
        """EI should return 3 impulses"""
        A, t = EI(omega_n=np.pi, zeta=0.02, Vtol=0.10)
        assert len(A) == 3, "EI should have 3 impulses"
        assert len(t) == 3, "EI should have 3 time points"
    
    def test_ei_unity_gain(self):
        """Sum of amplitudes should equal 1.0"""
        A, t = EI(omega_n=np.pi, zeta=0.02, Vtol=0.10)
        assert np.isclose(np.sum(A), 1.0, atol=1e-4), \
            f"Sum = {np.sum(A)}, expected 1.0"
    
    def test_ei_similar_duration_to_zvd(self):
        """EI duration should be similar to ZVD (both 3-impulse)"""
        omega_n = np.pi
        zeta = 0.02
        
        A_zvd, t_zvd, K = ZVD(omega_n, zeta)
        A_ei, t_ei = EI(omega_n, zeta, Vtol=0.10)
        
        # Should be within 20% of each other
        ratio = t_ei[-1] / t_zvd[-1]
        assert 0.8 < ratio < 1.2, \
            f"EI duration {t_ei[-1]} too different from ZVD {t_zvd[-1]}"
    
    def test_ei_all_amplitudes_positive(self):
        """All amplitudes should be positive"""
        A, t = EI(omega_n=np.pi, zeta=0.02, Vtol=0.10)
        assert np.all(A > 0), f"All amplitudes should be positive, got {A}"
    
    def test_ei_times_increasing(self):
        """Times should be monotonically increasing"""
        A, t = EI(omega_n=np.pi, zeta=0.02, Vtol=0.10)
        assert np.all(np.diff(t) > 0), f"Times should increase, got {t}"


class TestDesignShaper:
    """Tests for convenience function design_shaper()"""
    
    def test_design_shaper_zv(self):
        """design_shaper with method='ZV' should match ZV()"""
        omega_n = np.pi
        zeta = 0.02
        
        A1, t1, K = ZV(omega_n, zeta)
        A2, t2 = design_shaper(omega_n, zeta, method='ZV')
        
        assert np.allclose(A1, A2)
        assert np.allclose(t1, t2)
    
    def test_design_shaper_all_methods(self):
        """design_shaper should work for all methods"""
        omega_n = np.pi
        zeta = 0.02
        
        for method in ['ZV', 'ZVD', 'ZVDD', 'EI']:
            if method == 'EI':
                A, t = design_shaper(omega_n, zeta, method=method, Vtol=0.10, tol_band=0.20)
            else:
                A, t = design_shaper(omega_n, zeta, method=method)
            assert len(A) > 0
            assert len(t) > 0
            assert np.isclose(np.sum(A), 1.0, atol=1e-3)
    
    def test_design_shaper_case_insensitive(self):
        """Method name should be case insensitive"""
        omega_n = np.pi
        zeta = 0.02
        
        A1, t1 = design_shaper(omega_n, zeta, method='zvd')
        A2, t2 = design_shaper(omega_n, zeta, method='ZVD')
        A3, t3 = design_shaper(omega_n, zeta, method='ZvD')
        
        assert np.allclose(A1, A2)
        assert np.allclose(A1, A3)
    
    def test_design_shaper_invalid_method(self):
        """Invalid method should raise ValueError"""
        with pytest.raises(ValueError):
            design_shaper(omega_n=np.pi, zeta=0.02, method='INVALID')


class TestEdgeCases:
    """Tests for edge cases and error handling"""
    
    def test_zero_damping(self):
        """All shapers should handle zero damping"""
        omega_n = np.pi
        zeta = 0.0
        
        # Should not raise exceptions
        A_zv, t_zv, K = ZV(omega_n, zeta)
        A_zvd, t_zvd, K = ZVD(omega_n, zeta)
        A_zvdd, t_zvdd = ZVDD(omega_n, zeta)
    
    def test_very_low_frequency(self):
        """Shapers should handle very low frequencies (0.01 Hz)"""
        omega_n = 2 * np.pi * 0.01
        zeta = 0.02
        
        A_zv, t_zv, K = ZV(omega_n, zeta)
        assert t_zv[-1] > 10  # Should be long duration
        assert np.isclose(np.sum(A_zv), 1.0)
    
    def test_high_damping(self):
        """Shapers should handle high damping (ζ=0.5)"""
        omega_n = np.pi
        zeta = 0.5
        
        A_zv, t_zv, K = ZV(omega_n, zeta)
        A_zvd, t_zvd, K = ZVD(omega_n, zeta)
        
        assert np.isclose(np.sum(A_zv), 1.0)
        assert np.isclose(np.sum(A_zvd), 1.0)


def test_convolve_shapers_properties():
    omega1 = 2 * np.pi * 0.4
    omega2 = 2 * np.pi * 0.9
    A1, t1, _ = ZV(omega1, 0.02)
    A2, t2, _ = ZV(omega2, 0.02)

    A_conv, t_conv = convolve_shapers((A1, t1), (A2, t2))

    assert len(A_conv) == len(A1) * len(A2)
    assert np.isclose(np.sum(A_conv), 1.0, atol=1e-6)
    assert np.all(np.diff(t_conv) >= 0.0)


def test_design_multimode_cascaded_errors():
    with pytest.raises(ValueError):
        design_multimode_cascaded([0.4], [0.02, 0.02])

    with pytest.raises(ValueError):
        design_multimode_cascaded([], [])

    with pytest.raises(ValueError):
        design_multimode_cascaded([0.4], [0.02], method="INVALID")


def test_design_multimode_cascaded_two_modes():
    A, t = design_multimode_cascaded([0.4, 1.0], [0.02, 0.02], method="ZVD")

    assert len(A) == 9
    assert np.isclose(np.sum(A), 1.0, atol=1e-6)
    assert np.all(np.diff(t) >= 0.0)


def test_design_multimode_simultaneous_runs():
    with pytest.warns(UserWarning):
        A, t = design_multimode_simultaneous([0.4, 0.8], [0.02, 0.02], n_impulses=2, Vtol=0.2)

    assert len(A) == len(t) == 2
    assert np.isclose(np.sum(A), 1.0, atol=1e-2)


def test_get_shaper_info_contains_methods():
    info = shaper_module.get_shaper_info()

    assert "ZV" in info
    assert "ZVD" in info
    assert "ZVDD" in info
    assert "EI" in info


def test_design_multimode_simultaneous_warns_on_low_impulses():
    import warnings

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        design_multimode_simultaneous(
            [0.5, 0.8],
            [0.02, 0.02],
            n_impulses=2,
            Vtol=0.2,
        )

    assert any("too few" in str(w.message) for w in captured)


def test_design_multimode_simultaneous_warns_on_failed_optimization(monkeypatch):
    import warnings

    class DummyResult:
        success = False
        message = "forced failure"
        x = np.array([0.5, 0.5, 0.0, 1.0])

    def fake_minimize(*args, **kwargs):
        return DummyResult()

    monkeypatch.setattr(shaper_module, "minimize", fake_minimize)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        design_multimode_simultaneous(
            [0.5],
            [0.02],
            n_impulses=2,
            Vtol=0.2,
        )

    assert any("did not fully converge" in str(w.message) for w in captured)


# Run tests with: pytest tests/test_shapers.py -v
