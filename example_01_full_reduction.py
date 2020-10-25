from astropy.io import fits
from aspired import image_reduction
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

#
# Extract two spectra in a single science frame
#

# Science frame
lhs6328_frame = image_reduction.ImageReduction('sprat_LHS6328.list')
lhs6328_frame.reduce()
lhs6328_frame.inspect(
    filename='example_output/example_01_a_science_image',
    save_iframe=True)

lhs6328_twodspec = spectral_reduction.TwoDSpec(lhs6328_frame,
                                               cosmicray=True,
                                               readnoise=5.7)

lhs6328_twodspec.ap_trace(
    nspec=2,
    display=False,
    filename='example_output/example_01_a_science_aptrace',
    save_iframe=True)

lhs6328_twodspec.ap_extract(
    apwidth=15,
    skywidth=10,
    skydeg=1,
    optimal=True,
    display=False,
    filename='example_output/example_01_a_science_apextract',
    save_iframe=True)

lhs6328_twodspec.ap_extract(
    apwidth=15,
    skywidth=10,
    skydeg=1,
    optimal=True,
    forced=True,
    variances=lhs6328_twodspec.spectrum_list[1].var,
    display=False,
    filename='example_output/example_01_a_science_apextract_forced_weighted',
    save_iframe=True)

lhs6328_twodspec.ap_extract(
    apwidth=15,
    skywidth=10,
    skydeg=1,
    optimal=False,
    display=False,
    filename='example_output/example_01_a_science_apextract_tophat',
    save_iframe=True)

lhs6328_twodspec.ap_extract(
    apwidth=15,
    skywidth=10,
    skydeg=1,
    optimal=True,
    forced=True,
    variances=1000000.,
    display=False,
    filename=
    'example_output/example_01_a_science_apextract_forced_unit_weighted',
    save_iframe=True)

lhs6328_twodspec.save_fits(
    filename='example_output/example_01_a_science_traces',
    overwrite=True)

# Standard frame
standard_frame = image_reduction.ImageReduction('sprat_Hiltner102.list')
standard_frame.reduce()
standard_frame.inspect(
    filename='example_output/example_01_a_standard_image',
    save_iframe=True)

hilt102_twodspec = spectral_reduction.TwoDSpec(standard_frame,
                                               cosmicray=True,
                                               readnoise=5.7)

hilt102_twodspec.ap_trace(
    nspec=1,
    resample_factor=10,
    display=False,
    filename='example_output/example_01_a_standard_aptrace',
    save_iframe=True)

hilt102_twodspec.ap_extract(
    apwidth=15,
    skysep=3,
    skywidth=5,
    skydeg=1,
    optimal=True,
    display=False,
    filename='example_output/example_01_a_standard_apextract',
    save_iframe=True)

hilt102_twodspec.save_fits(
    filename='example_output/example_01_a_standard_trace',
    overwrite=True)

# Handle 1D Science spectrum
lhs6328_onedspec = spectral_reduction.OneDSpec()
lhs6328_onedspec.from_twodspec(lhs6328_twodspec, stype='science')
lhs6328_onedspec.from_twodspec(hilt102_twodspec, stype='standard')

# Add a 2D arc image
lhs6328_onedspec.add_arc(lhs6328_frame, stype='science')
lhs6328_onedspec.add_arc(standard_frame, stype='standard')

# Extract the 1D arc by aperture sum of the traces provided
lhs6328_onedspec.extract_arc_spec(
    display=False,
    stype='science+standard',
    filename='example_output/example_01_a_arc_spec',
    save_iframe=True)

# Find the peaks of the arc
lhs6328_onedspec.find_arc_lines(
    display=False,
    stype='science+standard',
    filename='example_output/example_01_a_arc_lines',
    save_iframe=True)

# Configure the wavelength calibrator
lhs6328_onedspec.initialise_calibrator(stype='science+standard')
lhs6328_onedspec.set_hough_properties(num_slopes=500,
                                      xbins=100,
                                      ybins=100,
                                      min_wavelength=3500,
                                      max_wavelength=8000,
                                      stype='science+standard')

lhs6328_onedspec.set_ransac_properties(filter_close=True,
                                       stype='science+standard')

lhs6328_onedspec.load_user_atlas(elements=element,
                                 wavelengths=atlas,
                                 stype='science+standard')
lhs6328_onedspec.do_hough_transform()

# Solve for the pixel-to-wavelength solution
lhs6328_onedspec.fit(max_tries=100, stype='science+standard', display=False)

# Apply the wavelength calibration and display it
lhs6328_onedspec.apply_wavelength_calibration(stype='science+standard')

#lhs6328_onedspec.wavecal_science.spectrum_list[0].create_wavelength_fits()

# Get the standard from the library
lhs6328_onedspec.load_standard(target='hiltner102')
lhs6328_onedspec.inspect_standard(
    save_iframe=True,
    filename='example_output/example_01_a_literature_standard')

lhs6328_onedspec.compute_sensitivity(kind='cubic', mask_fit_size=1)
lhs6328_onedspec.inspect_sensitivity(
    save_iframe=True,
    filename='example_output/example_01_a_sensitivity')

lhs6328_onedspec.apply_flux_calibration(stype='science+standard')

lhs6328_onedspec.inspect_reduced_spectrum(
    stype='science',
    save_iframe=True,
    filename='example_output/example_01_a_science_spectrum')

lhs6328_onedspec.inspect_reduced_spectrum(
    stype='standard',
    save_iframe=True,
    filename='example_output/example_01_a_standard_spectrum')

# Save as FITS
lhs6328_onedspec.save_fits(
    output='flux_resampled+wavecal+flux+count+trace',
    filename='example_output/example_01_a_full_reduction',
    stype='science+standard',
    overwrite=True)

# save as CSV
lhs6328_onedspec.save_csv(
    output='flux_resampled+wavecal+flux+count+trace',
    filename='example_output/example_01_a_full_reduction',
    stype='science+standard',
    overwrite=True)
