import os

import numpy as np
from aspired import spectral_reduction
from astropy.io import fits

HERE = os.getcwd()

# Load the image
lhs6328_fits = fits.open(
    os.path.join(
        HERE, "sprat_LHS6328_Hiltner102_raw", "v_e_20180810_12_1_0_2.fits.gz"
    )
)[1]
hilt102_fits = fits.open(
    os.path.join(
        HERE, "sprat_LHS6328_Hiltner102_raw", "v_s_20180810_27_1_0_2.fits.gz"
    )
)[1]

wave_bin = hilt102_fits.header["CDELT1"]
wave_start = hilt102_fits.header["CRVAL1"] + wave_bin / 2.0
wave_end = wave_start + (hilt102_fits.header["NAXIS1"] - 1) * wave_bin

wave = np.linspace(wave_start, wave_end, hilt102_fits.header["NAXIS1"])

#
# Loading two pre-saved spectral traces from a single FITS file.
#
lhs6328 = spectral_reduction.TwoDSpec(lhs6328_fits)
hilt102 = spectral_reduction.TwoDSpec(hilt102_fits)

# Trace the spectra
lhs6328.ap_trace(nspec=2, display=True)
hilt102.ap_trace(nspec=1, display=True)

# Extract the spectra
lhs6328.ap_extract(apwidth=10, optimal=True, skywidth=10, display=False)
hilt102.ap_extract(apwidth=15, optimal=True, skywidth=10, display=False)

# Calibrate the 1D spectra
lhs6328_onedspec = spectral_reduction.OneDSpec()
lhs6328_onedspec.from_twodspec(lhs6328, stype="science")
lhs6328_onedspec.from_twodspec(hilt102, stype="standard")

# Note that there are two science traces, so two wavelengths have to be
# supplied by in a list
lhs6328_onedspec.add_wavelength(wave, stype="science+standard")
lhs6328_onedspec.add_wavelength_resampled(wave, stype="science+standard")

# Get the standard flux from literature
lhs6328_onedspec.load_standard(target="hiltner102")

lhs6328_onedspec.get_sensitivity(k=3)
lhs6328_onedspec.inspect_sensitivity()
lhs6328_onedspec.apply_flux_calibration(stype="science")

# Inspect reduced spectrum
lhs6328_onedspec.inspect_reduced_spectrum(stype="science")

# Save as a FITS file
lhs6328_onedspec.save_fits(
    output="flux+wavecal_coefficients+count",
    filename=os.path.join(
        HERE,
        "..",
        "example_output",
        "example_03_wavelength_calibrated_science_and_standard_frames_only",
    ),
    stype="science",
    overwrite=True,
)
