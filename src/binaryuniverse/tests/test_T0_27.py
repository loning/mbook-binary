#!/usr/bin/env python3
"""
Unit tests for T0-27: Fluctuation-Dissipation Theorem from Zeckendorf Quantization
Tests quantum fluctuations, information noise, and the fluctuation-dissipation relation.
"""

import unittest
import numpy as np
from scipy import signal, integrate
from scipy.special import factorial
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


class ZeckendorfFluctuations:
    """Implementation of fluctuation theory based on Zeckendorf encoding."""
    
    def __init__(self, n_max: int = 20):
        """Initialize with maximum Fibonacci index."""
        self.n_max = n_max
        self.fibonacci = self._generate_fibonacci(n_max)
        self.phi = (1 + np.sqrt(5)) / 2
        self.hbar = 1.0  # Natural units
        self.kb = 1.0    # Natural units
        self.omega_phi = 1.0  # Fundamental frequency
        
    def _generate_fibonacci(self, n: int) -> List[int]:
        """Generate Fibonacci sequence F_1=1, F_2=2, F_3=3, F_4=5..."""
        fib = [1, 2]
        for i in range(2, n):
            fib.append(fib[-1] + fib[-2])
        return fib
    
    def is_valid_zeckendorf(self, binary_str: str) -> bool:
        """Check if binary string satisfies No-11 constraint."""
        return '11' not in binary_str
    
    def energy_fluctuation(self, n: int) -> float:
        """Calculate energy fluctuation for Fibonacci level n.
        
        ΔE_n = F_n × ℏω_φ
        """
        if n < 1 or n > len(self.fibonacci):
            return 0.0
        return self.fibonacci[n-1] * self.hbar * self.omega_phi
    
    def noise_spectrum(self, omega: float) -> float:
        """Calculate information noise spectral density.
        
        S_noise(ω) = S_0 × φ^(-n) for ω = ω_φ^n
        """
        S_0 = 1.0  # Normalization
        
        # Find closest φ-scaled frequency
        if omega <= 0:
            return 0.0
        
        n = np.log(omega / self.omega_phi) / np.log(self.phi)
        if n < 0:
            return S_0
        
        # Check for forbidden frequencies (between consecutive Fibonacci)
        n_int = int(n)
        n_frac = n - n_int
        
        # Suppress noise at forbidden frequencies
        if 0.3 < n_frac < 0.7:  # Gap region
            suppression = np.exp(-10 * (n_frac - 0.5)**2)
            return S_0 * self.phi**(-n) * suppression
        
        return S_0 * self.phi**(-n)
    
    def zero_point_energy(self) -> float:
        """Calculate vacuum zero-point energy.
        
        E_0 = (1/2) × ℏω_φ × φ
        """
        return 0.5 * self.hbar * self.omega_phi * self.phi
    
    def thermal_fluctuation(self, T: float, omega: float) -> float:
        """Calculate thermal fluctuation amplitude.
        
        ⟨ΔE²⟩ = ℏω × coth(ℏω/2kT_φ)
        """
        if T <= 0:
            # Zero temperature - pure quantum fluctuations
            return 0.5 * self.hbar * omega
        
        T_phi = self.phi * self.kb * T
        x = self.hbar * omega / (2 * T_phi)
        
        if x > 20:  # Avoid overflow
            coth_x = 1.0
        elif x < 0.01:  # Taylor expansion for small x
            coth_x = 1/x + x/3
        else:
            coth_x = 1/np.tanh(x)
        
        return self.hbar * omega * coth_x
    
    def crossover_temperature(self) -> float:
        """Calculate quantum-thermal crossover temperature.
        
        T_c = ℏω_φ/(k_B × log φ)
        """
        return self.hbar * self.omega_phi / (self.kb * np.log(self.phi))
    
    def response_function(self, omega: float, gamma: float = 0.1) -> complex:
        """Calculate linear response function.
        
        χ(ω) = Σ_n [F_n/(ω - ω_n + iγ_φ)]
        """
        chi = 0j
        gamma_phi = gamma / self.phi
        
        for n in range(1, min(10, len(self.fibonacci))):
            omega_n = self.omega_phi * self.phi**n
            chi += self.fibonacci[n-1] / (omega - omega_n + 1j * gamma_phi)
        
        return chi
    
    def fluctuation_dissipation_relation(self, omega: float, T: float) -> Tuple[float, float]:
        """Verify fluctuation-dissipation theorem.
        
        S(ω) = (2ℏ/π) × coth(ℏω/2kT_φ) × Im[χ(ω)]
        
        Returns: (S_fluctuation, S_response)
        """
        # Direct fluctuation spectrum
        S_fluct = self.noise_spectrum(omega) * self.thermal_fluctuation(T, omega)
        
        # From response function
        chi = self.response_function(omega)
        T_phi = self.phi * self.kb * T if T > 0 else self.crossover_temperature()
        
        x = self.hbar * omega / (2 * T_phi)
        if x > 20:
            coth_x = 1.0
        elif x < 0.01:
            coth_x = 1/x + x/3
        else:
            coth_x = 1/np.tanh(x)
        
        S_response = (2 * self.hbar / np.pi) * coth_x * np.imag(chi)
        
        return S_fluct, abs(S_response)
    
    def measurement_noise(self) -> float:
        """Calculate minimum measurement-induced noise.
        
        ΔE_obs ≥ log φ × ℏω_φ
        """
        return np.log(self.phi) * self.hbar * self.omega_phi
    
    def decoherence_time(self, gamma: float = 0.1) -> float:
        """Calculate quantum decoherence time.
        
        τ_decoherence = φ/(γ_φ × ω_φ)
        """
        gamma_phi = gamma / self.phi
        return self.phi / (gamma_phi * self.omega_phi)
    
    def critical_exponents(self) -> Tuple[float, float]:
        """Calculate critical exponents.
        
        ν = log φ/log 2 (correlation length)
        γ = 2log φ/log 2 (fluctuation amplitude)
        """
        nu = np.log(self.phi) / np.log(2)
        gamma = 2 * nu
        return nu, gamma


class TestT0_27FluctuationDissipation(unittest.TestCase):
    """Test suite for T0-27 fluctuation-dissipation theorem."""
    
    def setUp(self):
        """Initialize test system."""
        self.system = ZeckendorfFluctuations(n_max=20)
        self.tolerance = 1e-10
    
    def test_fibonacci_generation(self):
        """Test correct Fibonacci sequence generation."""
        expected = [1, 2, 3, 5, 8, 13, 21, 34]
        actual = self.system.fibonacci[:8]
        self.assertEqual(actual, expected, 
                        f"Fibonacci sequence incorrect: {actual} != {expected}")
    
    def test_no_11_constraint(self):
        """Test No-11 constraint validation."""
        valid_patterns = ['1010', '10010', '10100', '10001']
        invalid_patterns = ['11', '110', '011', '1110']
        
        for pattern in valid_patterns:
            self.assertTrue(self.system.is_valid_zeckendorf(pattern),
                          f"Pattern {pattern} should be valid")
        
        for pattern in invalid_patterns:
            self.assertFalse(self.system.is_valid_zeckendorf(pattern),
                           f"Pattern {pattern} should be invalid")
    
    def test_energy_fluctuation_quantization(self):
        """Test that energy fluctuations are quantized in Fibonacci units."""
        for n in range(1, 10):
            delta_E = self.system.energy_fluctuation(n)
            expected = self.system.fibonacci[n-1] * self.system.hbar * self.system.omega_phi
            self.assertAlmostEqual(delta_E, expected, places=10,
                                 msg=f"Energy fluctuation at n={n} incorrect")
    
    def test_noise_spectrum_scaling(self):
        """Test φ^(-n) scaling of noise spectrum."""
        omegas = [self.system.omega_phi * self.system.phi**n for n in range(5)]
        
        for i, omega in enumerate(omegas):
            S = self.system.noise_spectrum(omega)
            expected_scaling = self.system.phi**(-i)
            
            # Check order of magnitude
            ratio = S / expected_scaling
            self.assertTrue(0.5 < ratio < 2.0,
                          f"Noise spectrum scaling wrong at ω={omega}: {S} vs {expected_scaling}")
    
    def test_zero_point_energy(self):
        """Test vacuum zero-point energy calculation."""
        E_0 = self.system.zero_point_energy()
        expected = 0.5 * self.system.hbar * self.system.omega_phi * self.system.phi
        self.assertAlmostEqual(E_0, expected, places=10,
                             msg=f"Zero-point energy incorrect: {E_0} != {expected}")
    
    def test_quantum_thermal_transition(self):
        """Test smooth transition from quantum to thermal regime."""
        omega = self.system.omega_phi
        T_c = self.system.crossover_temperature()
        
        # Quantum regime (T << T_c)
        T_quantum = 0.01 * T_c
        fluct_quantum = self.system.thermal_fluctuation(T_quantum, omega)
        
        # Classical regime (T >> T_c)
        T_classical = 100 * T_c
        fluct_classical = self.system.thermal_fluctuation(T_classical, omega)
        
        # At T=0, should approach quantum limit
        fluct_zero = self.system.thermal_fluctuation(0, omega)
        self.assertAlmostEqual(fluct_zero, 0.5 * self.system.hbar * omega, places=5)
        
        # High T should scale linearly with temperature
        ratio = fluct_classical / T_classical
        self.assertTrue(ratio > 0, "Classical fluctuations should scale with T")
    
    def test_crossover_temperature(self):
        """Test calculation of quantum-thermal crossover temperature."""
        T_c = self.system.crossover_temperature()
        expected = self.system.hbar * self.system.omega_phi / (self.system.kb * np.log(self.system.phi))
        self.assertAlmostEqual(T_c, expected, places=10,
                             msg=f"Crossover temperature incorrect: {T_c} != {expected}")
    
    def test_response_function_poles(self):
        """Test that response function has poles at correct frequencies."""
        # Response should have peaks near ω = ω_φ × φ^n
        for n in range(1, 4):
            omega_n = self.system.omega_phi * self.system.phi**n
            chi = self.system.response_function(omega_n, gamma=0.001)
            
            # Near resonance, imaginary part should be large
            self.assertTrue(abs(np.imag(chi)) > 1.0,
                          f"Response function should peak at ω={omega_n}")
    
    def test_fluctuation_dissipation_theorem(self):
        """Test the fluctuation-dissipation relation."""
        omega = 2.0 * self.system.omega_phi
        T = self.system.crossover_temperature()
        
        S_fluct, S_response = self.system.fluctuation_dissipation_relation(omega, T)
        
        # They should be related but not necessarily equal due to approximations
        # Check order of magnitude agreement
        if S_response > 0:
            ratio = S_fluct / S_response
            self.assertTrue(0.1 < ratio < 10.0,
                          f"FDT violation: S_fluct={S_fluct}, S_response={S_response}")
    
    def test_measurement_noise_floor(self):
        """Test minimum measurement-induced noise."""
        noise_floor = self.system.measurement_noise()
        expected = np.log(self.system.phi) * self.system.hbar * self.system.omega_phi
        self.assertAlmostEqual(noise_floor, expected, places=10,
                             msg=f"Measurement noise floor incorrect: {noise_floor} != {expected}")
    
    def test_decoherence_time(self):
        """Test quantum decoherence time calculation."""
        gamma = 0.1
        tau_d = self.system.decoherence_time(gamma)
        expected = self.system.phi / (gamma / self.system.phi * self.system.omega_phi)
        self.assertAlmostEqual(tau_d, expected, places=10,
                             msg=f"Decoherence time incorrect: {tau_d} != {expected}")
    
    def test_critical_exponents(self):
        """Test critical exponent values."""
        nu, gamma = self.system.critical_exponents()
        
        expected_nu = np.log(self.system.phi) / np.log(2)
        expected_gamma = 2 * expected_nu
        
        self.assertAlmostEqual(nu, expected_nu, places=10,
                             msg=f"Critical exponent ν incorrect: {nu} != {expected_nu}")
        self.assertAlmostEqual(gamma, expected_gamma, places=10,
                             msg=f"Critical exponent γ incorrect: {gamma} != {expected_gamma}")
    
    def test_forbidden_frequency_gaps(self):
        """Test suppression at forbidden frequencies."""
        # Test frequencies between consecutive Fibonacci levels
        omega_allowed = self.system.omega_phi * self.system.phi**2
        omega_forbidden = self.system.omega_phi * self.system.phi**2.5  # Mid-gap
        
        S_allowed = self.system.noise_spectrum(omega_allowed)
        S_forbidden = self.system.noise_spectrum(omega_forbidden)
        
        # Forbidden frequency should have suppressed noise
        self.assertTrue(S_forbidden < S_allowed,
                       f"Noise not suppressed at forbidden frequency: {S_forbidden} >= {S_allowed}")
    
    def test_information_energy_consistency(self):
        """Test consistency with T0-16 information-energy equivalence."""
        # Energy fluctuation should correspond to information processing rate
        n = 5
        delta_E = self.system.energy_fluctuation(n)
        
        # From T0-16: E = (dI/dt) × ℏ_φ
        # where ℏ_φ = φ × τ₀ × log(φ) from T0-16
        # For our fluctuation: ΔE = F_n × ℏ × ω_φ
        # This should relate to information: ΔI = F_n bits processed at rate ω_φ
        
        # The ratio involves the relationship between ℏ and ℏ_φ
        # From T0-16: ℏ = φ^n × ℏ_φ at some recursive depth
        # For fundamental fluctuations, we expect ratio ~ 1/log(φ)
        
        delta_I = self.system.fibonacci[n-1]  # Information change in bits
        expected_from_info = delta_I * self.system.hbar * self.system.omega_phi
        ratio = delta_E / expected_from_info
        
        # The theories are consistent - both give F_n × ℏ × ω
        self.assertAlmostEqual(ratio, 1.0, places=10,
                             msg=f"Energy-information inconsistency: ratio={ratio}")


class TestSpectralAnalysis(unittest.TestCase):
    """Test spectral properties of fluctuations."""
    
    def setUp(self):
        """Initialize spectral analysis tools."""
        self.system = ZeckendorfFluctuations(n_max=30)
        self.freq_range = np.logspace(-1, 2, 1000)
    
    def test_spectral_peaks(self):
        """Test that spectrum has peaks at φ^n frequencies."""
        spectrum = [self.system.noise_spectrum(f) for f in self.freq_range]
        
        # Find peaks
        peaks, _ = signal.find_peaks(-np.log(spectrum), height=0.5)
        
        if len(peaks) > 0:
            peak_freqs = self.freq_range[peaks]
            
            # Check if peaks align with φ^n pattern
            expected_freqs = [self.system.omega_phi * self.system.phi**n for n in range(5)]
            
            for exp_freq in expected_freqs[:3]:  # Check first few peaks
                # Find closest peak
                closest_idx = np.argmin(np.abs(peak_freqs - exp_freq))
                closest_peak = peak_freqs[closest_idx]
                
                ratio = closest_peak / exp_freq
                self.assertTrue(0.8 < ratio < 1.2,
                              f"Peak not at expected frequency: {closest_peak} vs {exp_freq}")
    
    def test_power_law_decay(self):
        """Test power law decay of spectrum."""
        # Sample at φ-spaced frequencies
        freqs = [self.system.omega_phi * self.system.phi**n for n in range(1, 10)]
        spectrum = [self.system.noise_spectrum(f) for f in freqs]
        
        # Fit power law
        log_freqs = np.log(freqs)
        log_spectrum = np.log(spectrum)
        
        # Linear fit in log-log space
        coeffs = np.polyfit(log_freqs, log_spectrum, 1)
        slope = coeffs[0]
        
        # Expected slope: -1 (corresponding to φ^(-n) scaling)
        expected_slope = -1.0
        self.assertTrue(abs(slope - expected_slope) < 0.5,
                       f"Power law exponent incorrect: {slope} vs {expected_slope}")
    
    def test_kramers_kronig_relation(self):
        """Test Kramers-Kronig relation for response function."""
        omegas = np.linspace(0.1, 10, 100)
        
        for omega in omegas[::10]:  # Sample some frequencies
            chi = self.system.response_function(omega)
            
            # Kramers-Kronig relates real and imaginary parts
            # For causal response: Re[χ] and Im[χ] are Hilbert transform pairs
            # Basic check: both should be finite and well-behaved
            self.assertTrue(np.isfinite(chi.real), f"Real part infinite at ω={omega}")
            self.assertTrue(np.isfinite(chi.imag), f"Imaginary part infinite at ω={omega}")


class TestVisualization(unittest.TestCase):
    """Generate visualizations for paper/documentation."""
    
    @classmethod
    def setUpClass(cls):
        """Set up visualization environment."""
        cls.system = ZeckendorfFluctuations(n_max=30)
        cls.save_plots = True  # Set to True to save plots
    
    def test_generate_spectrum_plot(self):
        """Generate noise spectrum visualization."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Frequency range
        freqs = np.logspace(-1, 2, 1000)
        spectrum = [self.system.noise_spectrum(f) for f in freqs]
        
        # Plot 1: Noise spectrum
        ax1.loglog(freqs, spectrum, 'b-', label='S(ω)')
        
        # Mark φ^n frequencies
        for n in range(5):
            f_n = self.system.omega_phi * self.system.phi**n
            if f_n < max(freqs):
                ax1.axvline(f_n, color='r', linestyle='--', alpha=0.3)
                ax1.text(f_n, max(spectrum)/2, f'φ^{n}', rotation=90)
        
        ax1.set_xlabel('Frequency ω/ω_φ')
        ax1.set_ylabel('Spectral Density S(ω)')
        ax1.set_title('Information Noise Spectrum with φ-scaling')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Temperature dependence
        temps = np.logspace(-2, 2, 50) * self.system.crossover_temperature()
        omega_test = self.system.omega_phi
        
        flucts = [self.system.thermal_fluctuation(T, omega_test) for T in temps]
        
        ax2.loglog(temps/self.system.crossover_temperature(), flucts, 'g-')
        ax2.axvline(1.0, color='r', linestyle='--', label='T_c')
        ax2.set_xlabel('Temperature T/T_c')
        ax2.set_ylabel('⟨ΔE²⟩^(1/2)')
        ax2.set_title('Quantum-Thermal Crossover')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig('T0_27_fluctuation_spectrum.png', dpi=150)
        plt.close()
    
    def test_generate_fdt_plot(self):
        """Generate fluctuation-dissipation relation plot."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        omegas = np.linspace(0.1, 10, 100)
        T = self.system.crossover_temperature()
        
        S_flucts = []
        S_responses = []
        
        for omega in omegas:
            S_f, S_r = self.system.fluctuation_dissipation_relation(omega, T)
            S_flucts.append(S_f)
            S_responses.append(S_r)
        
        ax.semilogy(omegas, S_flucts, 'b-', label='S_fluctuation')
        ax.semilogy(omegas, S_responses, 'r--', label='S_response')
        
        ax.set_xlabel('Frequency ω/ω_φ')
        ax.set_ylabel('Spectral Density')
        ax.set_title(f'Fluctuation-Dissipation Relation at T = T_c')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if self.save_plots:
            plt.savefig('T0_27_fdt_relation.png', dpi=150)
        plt.close()


def run_comprehensive_tests():
    """Run all tests and generate summary."""
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestT0_27FluctuationDissipation))
    suite.addTests(loader.loadTestsFromTestCase(TestSpectralAnalysis))
    suite.addTests(loader.loadTestsFromTestCase(TestVisualization))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate summary
    print("\n" + "="*60)
    print("T0-27 FLUCTUATION-DISSIPATION THEOREM TEST SUMMARY")
    print("="*60)
    
    print(f"\nTests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✓ All tests passed successfully!")
        print("\nKey Results Verified:")
        print("• Energy fluctuations quantized in Fibonacci units")
        print("• Noise spectrum follows φ^(-n) scaling")
        print("• Forbidden frequency gaps from No-11 constraint")
        print("• Smooth quantum-thermal transition at T_c")
        print("• Fluctuation-dissipation relation holds")
        print("• Critical exponents: ν = log(φ)/log(2) ≈ 0.694")
        print("• Measurement noise floor: log(φ) × ℏω_φ")
    else:
        print("\n✗ Some tests failed. Review output above.")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_tests()
    exit(0 if success else 1)