import numpy as np
from aspired import image_reduction
from aspired import spectral_reduction

# Line list
wavelengths = [
    4193.5, 4385.77, 4500.98, 4524.68, 4582.75, 4624.28, 4671.23, 4697.02,
    4734.15, 4807.02, 4921.48, 5028.28, 5618.88, 5823.89, 5893.29, 5934.17,
    6182.42, 6318.06, 6472.841, 6595.56, 6668.92, 6728.01, 6827.32, 6976.18,
    7119.60, 7257.9, 7393.8, 7584.68, 7642.02, 7740.31, 7802.65, 7887.40,
    7967.34, 8057.258
]
elements = ['Xe'] * len(wavelengths)

spatial_mask = np.arange(50, 220)
spec_mask = np.arange(50, 1024)

#
# Extract two spectra in a single science frame
#

# Science frame
lhs6328_frame = image_reduction.ImageReduction(log_file_name=None,
                                               log_level='INFO')
lhs6328_frame.add_filelist('example_use_cases/sprat_LHS6328.list')
lhs6328_frame.load_data()
lhs6328_frame.reduce()

lhs6328_twodspec = spectral_reduction.TwoDSpec(lhs6328_frame,
                                               spatial_mask=spatial_mask,
                                               spec_mask=spec_mask,
                                               cosmicray=True,
                                               readnoise=5.7,
                                               gain=2.45,
                                               sigclip=0.1,
                                               psfsize=11.0,
                                               psffwhm=1.5,
                                               psfmodel='gaussy',
                                               fsmode='convolve',
                                               cleantype='medmask',
                                               log_file_name=None,
                                               log_level='INFO')

lhs6328_twodspec.ap_trace(
    nspec=2,
    display=True,
    filename='example_output/example_01_a_science_aptrace',
    save_fig=True)

# Optimal extraction to get the LSF for force extraction below
lhs6328_twodspec.ap_extract(
    apwidth=15,
    skywidth=10,
    skydeg=1,
    optimal=True,
    display=True,
    filename='example_output/example_01_a_science_apextract',
    save_fig=True)

# Standard frame
standard_frame = image_reduction.ImageReduction(log_file_name=None)
standard_frame.add_filelist('example_use_cases/sprat_Hiltner102.list')
standard_frame.load_data()
standard_frame.reduce()

hilt102_twodspec = spectral_reduction.TwoDSpec(standard_frame,
                                               cosmicray=True,
                                               spatial_mask=spatial_mask,
                                               spec_mask=spec_mask,
                                               readnoise=5.7,
                                               gain=2.6,
                                               log_file_name=None,
                                               log_level='INFO')

hilt102_twodspec.ap_trace(
    nspec=1,
    resample_factor=10,
    display=False,
    filename='example_output/example_01_a_standard_aptrace',
    save_fig=True)

hilt102_twodspec.ap_extract(
    apwidth=15,
    skysep=3,
    skywidth=5,
    skydeg=1,
    optimal=True,
    display=False,
    filename='example_output/example_01_a_standard_apextract',
    save_fig=True)

# Extract the 1D arc by aperture sum of the traces provided
lhs6328_twodspec.extract_arc_spec(
    display=False,
    filename='example_output/example_01_a_arc_spec_science',
    save_fig=True)

hilt102_twodspec.extract_arc_spec(
    display=False,
    filename='example_output/example_01_a_arc_spec_standard',
    save_fig=True)

# Handle 1D Science spectrum
lhs6328_onedspec = spectral_reduction.OneDSpec(log_file_name=None,
                                               log_level='INFO')
lhs6328_onedspec.from_twodspec(lhs6328_twodspec, stype='science')
lhs6328_onedspec.from_twodspec(hilt102_twodspec, stype='standard')

# Find the peaks of the arc
lhs6328_onedspec.find_arc_lines(
    prominence=5.,
    display=False,
    stype='science+standard',
    filename='example_output/example_01_a_arc_lines',
    save_fig=True)

# Configure the wavelength calibrator
lhs6328_onedspec.initialise_calibrator(stype='science+standard')
lhs6328_onedspec.set_hough_properties(xbins=100,
                                      ybins=100,
                                      min_wavelength=3500,
                                      max_wavelength=8200,
                                      stype='science+standard')

lhs6328_onedspec.set_ransac_properties(filter_close=True,
                                       ransac_tolerance=1,
                                       stype='science+standard')

lhs6328_onedspec.add_user_atlas(elements=elements,
                                wavelengths=wavelengths,
                                constrain_poly=True,
                                stype='science+standard')

# Solve for the pixel-to-wavelength solution
lhs6328_onedspec.do_hough_transform(brute_force=False)
lhs6328_onedspec.fit(stype='science+standard', display=True)

# Apply the wavelength calibration and display it
lhs6328_onedspec.apply_wavelength_calibration(stype='science+standard')

# Get the standard from the library
lhs6328_onedspec.load_standard(target='hiltner102')

lhs6328_onedspec.compute_sensitivity(k=3, mask_fit_size=1)
lhs6328_onedspec.inspect_sensitivity(
    save_fig=True, filename='example_output/example_01_a_sensitivity')

lhs6328_onedspec.apply_flux_calibration(stype='science+standard')

lhs6328_onedspec.inspect_reduced_spectrum(
    stype='science',
    save_fig=True,
    filename='example_output/example_01_a_science_spectrum')
