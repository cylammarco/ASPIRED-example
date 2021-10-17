import numpy as np
from aspired import image_reduction
from aspired import spectral_reduction

# pixel-wavelength mapping
pix = np.arange(10) * 100
wave = np.array((3678.27600444, 4099.95638482, 4536.85629523, 4986.55350976,
                 5446.64818767, 5914.76287336, 6388.54249641, 6865.65437158,
                 7343.78819876, 7820.65606301))

pw_coeff = np.polynomial.polynomial.polyfit(pix, wave, deg=4)

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
                                               psfmodel='gaussy',
                                               fsmode='convolve',
                                               cleantype='medmask',
                                               log_file_name=None,
                                               log_level='INFO')

lhs6328_twodspec.ap_trace(
    nspec=2,
    display=True,
    filename='example_output/example_08_a_science_aptrace',
    save_fig=True)

# Optimal extraction to get the LSF for force extraction below
lhs6328_twodspec.ap_extract(
    apwidth=15,
    skywidth=10,
    skydeg=1,
    optimal=True,
    display=True,
    filename='example_output/example_08_a_science_apextract',
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
    filename='example_output/example_08_a_standard_aptrace',
    save_fig=True)

hilt102_twodspec.ap_extract(
    apwidth=15,
    skysep=3,
    skywidth=5,
    skydeg=1,
    optimal=True,
    display=False,
    filename='example_output/example_08_a_standard_apextract',
    save_fig=True)

# Extract the 1D arc by aperture sum of the traces provided
lhs6328_twodspec.extract_arc_spec(
    display=False,
    filename='example_output/example_08_a_arc_spec_science',
    save_fig=True)

hilt102_twodspec.extract_arc_spec(
    display=False,
    filename='example_output/example_08_a_arc_spec_standard',
    save_fig=True)

# Handle 1D Science spectrum
lhs6328_onedspec = spectral_reduction.OneDSpec(log_file_name=None,
                                               log_level='INFO')
lhs6328_onedspec.from_twodspec(lhs6328_twodspec, stype='science')
lhs6328_onedspec.from_twodspec(hilt102_twodspec, stype='standard')

lhs6328_onedspec.initialise_calibrator(stype='science+standard')

# Find the peaks of the arc
lhs6328_onedspec.add_fit_coeff(pw_coeff)

# Solve for the pixel-to-wavelength solution
lhs6328_onedspec.fit(stype='science+standard', display=True)

# Apply the wavelength calibration and display it
lhs6328_onedspec.apply_wavelength_calibration(stype='science+standard')

# Get the standard from the library
lhs6328_onedspec.load_standard(target='hiltner102')

lhs6328_onedspec.get_sensitivity(k=3, mask_fit_size=1)
lhs6328_onedspec.inspect_sensitivity(
    save_fig=True, filename='example_output/example_08_a_sensitivity')

lhs6328_onedspec.apply_flux_calibration(stype='science+standard')

lhs6328_onedspec.inspect_reduced_spectrum(
    stype='science',
    save_fig=True,
    filename='example_output/example_08_a_science_spectrum')
