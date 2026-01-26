"""
Guide for Using PyFocus to Create PSFs Matching MTF Targets

PyFocus is a library for vectorial calculations of focused optical fields.
It's primarily designed for microscopy applications.

Installation:
    pip install PyCustomFocus

Note: PyCustomFocus requires Python 3.6+ and several dependencies.
"""

import matplotlib

# matplotlib.use("Agg")
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from PyFocus import sim


class PyFocusMTFMatcher:
    """Use PyFocus to create realistic PSFs matching MTF targets."""

    def __init__(self, size=512, wire_fov_mm=50.0):

        self.size = size
        self.wire_fov_mm = wire_fov_mm
        self.half = size // 2
        self.dx = wire_fov_mm / size

        # Setup spatial coordinates
        x = np.linspace(-self.half, self.half, size) * self.dx
        y = np.linspace(-self.half, self.half, size) * self.dx
        self.X, self.Y = np.meshgrid(x, y)
        self.R = np.sqrt(self.X**2 + self.Y**2)

        # Setup frequency coordinates
        fx = np.fft.fftshift(np.fft.fftfreq(size, d=self.dx))
        fy = np.fft.fftshift(np.fft.fftfreq(size, d=self.dx))
        Fx, Fy = np.meshgrid(fx, fy)
        self.F = np.sqrt(Fx**2 + Fy**2)
        self.f_rad = self.F[self.half, self.half :]

    def compute_mtf_from_psf(self, psf):
        """Compute MTF from PSF."""
        psf = psf / psf.sum()
        FT = np.fft.fft2(np.fft.ifftshift(psf))
        mtf2d = np.fft.fftshift(np.abs(FT))
        mtf2d /= np.max(mtf2d)
        mtf_rad = mtf2d[self.half, self.half :]
        return interp1d(
            self.f_rad, mtf_rad, kind="linear", bounds_error=False, fill_value=0.0
        )

    def create_pyfocus_psf(
        self,
        patient_fov,
        pixel_spacing,
        na,
        n=1.33,
        h=3,
        w0=5.0,
        wavelength=550,
        gamma=45,
        beta=90,
        z=-1,
        z_steps=1,
        z_range=2,
        I0=1,
    ):
        """

        Args:
            patient_fov (float): Field of view in mm
            pixel_spacing (float): Pixel spacing in mm
            na (float): Numerical aperture (typically 0.1 to 1.4)
            n (float, optional): Refraction index for the medium of the optical system. Defaults to 1.33.
            h (int, optional): Radius of aperture of the objective lens in mm. Defaults to 3.
            w0 (float, optional): Radius of the incident gaussian beam in mm. Defaults to 5.0.
            wavelength (int, optional): Wavelength in vacuum in nm. Defaults to 550.
            gamma (int, optional): Parameter that determines the polarization, arctan(ey/ex) (gamma=45, beta=90 gives right circular polarization and a donut shape). Defaults to 45.
            beta (int, optional): Parameter that determines the polarization, phase difference between ex and ey (gamma=45, beta=90 gives right circular polarization and a donut shape). Defaults to 90.
            z (int, optional): Axial position for the XY plane in nm. Defaults to 0.
            z_steps (int, optional): Resolution in the axial coordinate (Z) for the focused field in nm. Defaults to 1.
            z_range (int, optional): Field of view in the axial coordinate (Z) in which the focused field is calculated in nm. Defaults to 2.
            I0 (int, optional): Incident field intensity in mW/cm^2. Defaults to 1.

        Returns:
            _type_: _description_
        """

        try:
            # Calculate focused field using PyFocus
            # This uses the no_mask function for an aberration-free system

            # Resolution in the X or Y coordinate for the focused field in nm
            x_steps = pixel_spacing

            # Field of view in the X or Y coordinate in which the focused field is calculated in nm
            x_range = patient_fov + (x_steps * 2)
            na = 1.4
            n = 1.7
            h = 0.3
            w0 = 0.5
            wavelength = 32
            I0 = 1
            z = -1
            gamma = 45
            beta = 60
            L = ""
            R = ""
            ds = ""
            z_int = ""
            figure_name = ""
            parameters = np.array(
                (
                    na,
                    n,
                    h,
                    w0,
                    wavelength,
                    gamma,
                    beta,
                    z,
                    x_steps,
                    z_steps,
                    x_range,
                    z_range,
                    I0,
                    L,
                    R,
                    ds,
                    z_int,
                    figure_name,
                ),
                dtype=object,
            )
            ex_XZ, ey_XZ, ez_XZ, ex_XY, ey_XY, ez_XY = sim.VP(False, False, *parameters)
            # Calculate intensity (PSF)
            intensity = np.abs(ex_XY) ** 2 + np.abs(ey_XY) ** 2 + np.abs(ez_XY) ** 2

            # Normalize
            psf = intensity / intensity.sum()
            psf = psf / psf.sum()
            FT = np.fft.fft2(np.fft.ifftshift(psf))
            mtf2d = np.fft.fftshift(np.abs(FT))
            mtf2d /= np.max(mtf2d)
            mtf_rad = mtf2d[self.half, self.half :]
            mtfc_fun = interp1d(
                self.f_rad,
                mtf_rad,
                kind="linear",
                bounds_error=False,
                fill_value=0.0,
            )
            x_vals = np.linspace(0, 1, 100)
            y_vals = mtfc_fun(x_vals)
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(psf, cmap="hot")
            plt.colorbar()
            plt.title("Point Spread Function (PSF)")
            plt.xlabel("Pixel")
            plt.ylabel("Pixel")
            plt.subplot(1, 2, 2)
            plt.plot(x_vals, y_vals)
            plt.title("Modulation Transfer Function (MTF)")
            plt.xlabel("Normalized Frequency (cycles/pixel)")
            plt.ylabel("MTF")
            plt.show()

            return psf

        except Exception as e:
            print(f"Error in PyFocus calculation: {e}")
            print("Falling back to Gaussian approximation...")
            # Fallback: approximate PSF as Gaussian
            # Airy disk radius ≈ 0.61 * lambda / na
            sigma_m = 0.61 * wavelength / na / 2.355  # Convert FWHM to sigma
            sigma_mm = sigma_m * 1e3
            psf = np.exp(-self.R**2 / (2 * sigma_mm**2))
            return psf / psf.sum()

    def optimize_for_mtf(
        self, patient_fov, pixel_spacing, mtf50, mtf10, wavelength_nm=550, fixed_n=1.0
    ):
        """
        Optimize PyFocus parameters to match MTF targets.

        Parameters:
        -----------
        mtf50 : float
            Frequency where MTF = 0.5 (cycles/mm)
        mtf10 : float
            Frequency where MTF = 0.1 (cycles/mm)
        wavelength_nm : float
            Wavelength in nm
        fixed_n : float
            Refractive index (fixed)

        Returns:
        --------
        dict : Results with optimal PSF and parameters
        """
        target_freqs = np.array([mtf50, mtf10])
        target_mtf_values = np.array([0.5, 0.1])

        def objective(params):
            """Objective function for optimization."""
            na = params[0]

            try:
                # Create PSF with current parameters
                psf = self.create_pyfocus_psf(
                    patient_fov,
                    pixel_spacing,
                    na,
                )

                # Compute MTF
                mtf_func = self.compute_mtf_from_psf(psf)

                # Evaluate at target frequencies
                predicted_mtf = np.array([mtf_func(f) for f in target_freqs])

                # Compute error
                error = np.sum((predicted_mtf - target_mtf_values) ** 2)
                print(f"Current error: {error:.6e}")

                return error

            except Exception as e:
                print(f"Error during optimization: {e}")
                return 1e10  # Large penalty

        # Setup optimization
        initial_guess = [0.5]  # NA only
        bounds = [(0.1, 1.4)]
        param_names = ["NA"]

        print("Starting optimization...")
        print(f"Initial guess: {dict(zip(param_names, initial_guess))}")

        # Optimize
        result = minimize(
            objective,
            initial_guess,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 50, "disp": True},
        )

        print(f"\nOptimization complete!")
        print(f"Success: {result.success}")
        print(f"Final error: {result.fun:.6e}")

        # Extract parameters
        optimal_params = dict(zip(param_names, result.x))
        print(f"Optimal parameters: {optimal_params}")

        # Create final PSF
        final_psf = self.create_pyfocus_psf(patient_fov, pixel_spacing, na=result.x[0])

        # Compute final MTF
        final_mtf = self.compute_mtf_from_psf(final_psf)

        # Verify
        achieved_mtf = [final_mtf(f) for f in target_freqs]
        print(f"\nVerification:")
        print(f"Target MTF: {target_mtf_values}")
        print(f"Achieved MTF: {np.array(achieved_mtf)}")
        print(f"Errors: {np.array(achieved_mtf) - target_mtf_values}")

        return {
            "psf": final_psf,
            "mtf_function": final_mtf,
            "params": optimal_params,
            "wavelength_nm": wavelength_nm,
            "n": fixed_n,
            "target_freqs": target_freqs,
            "achieved_mtf": achieved_mtf,
            "optimization_result": result,
        }

    def plot_results(self, results):
        """Plot PSF and MTF results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        psf = results["psf"]
        extent = [
            -self.wire_fov_mm / 2,
            self.wire_fov_mm / 2,
            -self.wire_fov_mm / 2,
            self.wire_fov_mm / 2,
        ]

        # PSF 2D
        im = axes[0, 0].imshow(psf, extent=extent, cmap="hot", origin="lower")
        axes[0, 0].set_xlabel("X (mm)")
        axes[0, 0].set_ylabel("Y (mm)")
        title = f"PSF (NA={results['params'].get('NA', 'N/A'):.3f})"
        axes[0, 0].set_title(title)
        plt.colorbar(im, ax=axes[0, 0])

        # PSF log scale
        im2 = axes[0, 1].imshow(
            np.log10(psf + 1e-10), extent=extent, cmap="hot", origin="lower"
        )
        axes[0, 1].set_xlabel("X (mm)")
        axes[0, 1].set_ylabel("Y (mm)")
        axes[0, 1].set_title("PSF (log scale)")
        plt.colorbar(im2, ax=axes[0, 1])

        # Radial profile
        axes[1, 0].plot(self.R[self.half, :], psf[self.half, :], "b-", linewidth=2)
        axes[1, 0].set_xlabel("Radius (mm)")
        axes[1, 0].set_ylabel("Intensity")
        axes[1, 0].set_title("PSF Radial Profile")
        axes[1, 0].grid(alpha=0.3)
        axes[1, 0].set_xlim(0, 5)

        # MTF
        freqs = np.linspace(0, 1.5, 1000)
        mtf_values = [results["mtf_function"](f) for f in freqs]
        axes[1, 1].plot(freqs, mtf_values, "b-", linewidth=2, label="Achieved")
        axes[1, 1].plot(
            results["target_freqs"],
            [0.5, 0.1],
            "ro",
            markersize=10,
            label="Targets",
            zorder=5,
        )
        axes[1, 1].axhline(0.5, color="r", linestyle="--", alpha=0.3)
        axes[1, 1].axhline(0.1, color="r", linestyle="--", alpha=0.3)
        axes[1, 1].set_xlabel("Spatial Frequency (cycles/mm)")
        axes[1, 1].set_ylabel("MTF")
        axes[1, 1].set_title("MTF Curve")
        axes[1, 1].grid(alpha=0.3)
        axes[1, 1].legend()
        axes[1, 1].set_xlim(0, 1.5)
        axes[1, 1].set_ylim(0, 1.05)

        plt.tight_layout()
        return fig


# Example usage
if __name__ == "__main__":
    # Target MTF points
    mtf50 = 0.434  # cycles/mm
    mtf10 = 0.730  # cycles/mm

    print("=" * 70)
    print("PyFocus PSF-MTF Matching")
    print("=" * 70)
    print(f"Target: MTF = 0.5 at {mtf50} cycles/mm")
    print(f"Target: MTF = 0.1 at {mtf10} cycles/mm\n")

    # Create matcher
    matcher = PyFocusMTFMatcher(size=512, wire_fov_mm=50.0)
    patient_fov = 340  # mm
    pixel_spacing = 0.6640625  # mm

    # Optimize (NA only, faster)
    print("Optimizing NA only...")
    results = matcher.optimize_for_mtf(
        patient_fov,
        pixel_spacing,
        mtf50=mtf50,
        mtf10=mtf10,
        wavelength_nm=550,
        fixed_n=1.0,
    )

    # Plot
    fig = matcher.plot_results(results)
    plt.savefig("pyfocus_results.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved as 'pyfocus_results.png'")

    # Save PSF
    np.save("pyfocus_psf.npy", results["psf"])
    print("PSF saved as 'pyfocus_psf.npy'")

    plt.show()

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
