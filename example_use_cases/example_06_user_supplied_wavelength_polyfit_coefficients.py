import numpy as np
from astropy.io import fits
from aspired import spectral_reduction

# Load the image
lhs6328_fits = fits.open(
    'example_use_cases/sprat_LHS6328_Hiltner102_raw/v_e_20180810_12_1_0_0.fits.gz')[0]
spatial_mask = np.arange(50, 200)
spec_mask = np.arange(50, 1024)

#
# Loading two pre-saved spectral traces from a single FITS file.
#
lhs6328 = spectral_reduction.TwoDSpec(lhs6328_fits,
                                      spatial_mask=spatial_mask,
                                      spec_mask=spec_mask,
                                      cosmicray=True,
                                      readnoise=2.34)

# Trace the spectra
lhs6328.ap_trace(nspec=2, display=False)

# Extract the spectra
lhs6328.ap_extract(apwidth=10, optimal=True, skywidth=10, display=False)

# Calibrate the 1D spectra
lhs6328_onedspec = spectral_reduction.OneDSpec()
lhs6328_onedspec.from_twodspec(lhs6328, stype='science')

fit_coeff =np.array([
        3.09833375e+03, 5.98842823e+00, -2.83963934e-03, 2.84842392e-06,
        -1.03725267e-09
    ])
fit_type = 'poly'

# Note that there are two science traces, so two polyfit coefficients have to
# be supplied by in a list
lhs6328_onedspec.add_fit_coeff(fit_coeff, fit_type, stype='science')
lhs6328_onedspec.apply_wavelength_calibration(stype='science')

# Inspect reduced spectrum
lhs6328_onedspec.inspect_reduced_spectrum(display=True, stype='science')

# Save as a FITS file
lhs6328_onedspec.save_fits(
    output='wavecal+count',
    filename=
    'example_output/example_06_user_supplied_wavelength_polyfit_coefficients',
    stype='science',
    overwrite=True)
