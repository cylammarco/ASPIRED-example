from astropy.io import fits
from aspired import spectral_reduction
import numpy as np

# Line list
atlas = [
    4193.5, 4385.77, 4500.98, 4524.68, 4582.75, 4624.28, 4671.23, 4697.02,
    4734.15, 4807.02, 4921.48, 5028.28, 5618.88, 5823.89, 5893.29, 5934.17,
    6182.42, 6318.06, 6472.841, 6595.56, 6668.92, 6728.01, 6827.32, 6976.18,
    7119.60, 7257.9, 7393.8, 7584.68, 7642.02, 7740.31, 7802.65, 7887.40,
    7967.34, 8057.258
]
element = ['Xe'] * len(atlas)

spatial_mask = np.arange(50, 200)
spec_mask = np.arange(50, 1024)

# Load the arc
arc_fits = fits.open(
    'example_use_cases/sprat_LHS6328_Hiltner102_raw/v_a_20180810_13_1_0_1.fits.gz')[0]

#
# Loading pre-saved spectral traces from a single FITS file.
#
lhs6328_extracted = fits.open(
    'example_output/example_01_a_full_reduction_science_0.fits')
lhs6328_trace = lhs6328_extracted[1].data
lhs6328_trace_sigma = lhs6328_extracted[2].data
lhs6328_adu = lhs6328_extracted[3].data

lhs6328_twodspec = spectral_reduction.TwoDSpec(spatial_mask=spatial_mask,
                                               spec_mask=spec_mask)

# Add a 2D arc image
lhs6328_twodspec.add_arc(arc_fits)
lhs6328_twodspec.apply_twodspec_mask_to_arc()

# Add the trace and the line spread function (sigma) to the 2D arc image
lhs6328_twodspec.add_trace(lhs6328_trace, lhs6328_trace_sigma)

# Extract the 1D arc by aperture sum of the traces provided
lhs6328_twodspec.extract_arc_spec(display=False)

lhs6328_onedspec = spectral_reduction.OneDSpec()
lhs6328_onedspec.from_twodspec(lhs6328_twodspec)

# Find the peaks of the arc
lhs6328_onedspec.find_arc_lines(display=False, stype='science')

# Configure the wavelength calibrator
lhs6328_onedspec.initialise_calibrator()
lhs6328_onedspec.set_hough_properties(num_slopes=500,
                                      xbins=100,
                                      ybins=100,
                                      min_wavelength=3500,
                                      max_wavelength=8000)
lhs6328_onedspec.add_user_atlas(wavelengths=atlas,
                                 elements=element,
                                 stype='science')
lhs6328_onedspec.set_ransac_properties()
lhs6328_onedspec.do_hough_transform()

# Solve for the pixel-to-wavelength solution
lhs6328_onedspec.fit(max_tries=1000, stype='science')

# Add the extracted 1D spectrum without the uncertainties and sky
lhs6328_onedspec.add_spec(lhs6328_adu, stype='science')

# Apply the wavelength calibration and display it
lhs6328_onedspec.apply_wavelength_calibration(stype='science')
lhs6328_onedspec.inspect_reduced_spectrum(stype='science')

# Save as a FITS file
lhs6328_onedspec.save_fits(
    output='wavecal+count',
    filename='example_output/example_02_wavelength_calibrated_spectrum',
    stype='science',
    overwrite=True)
