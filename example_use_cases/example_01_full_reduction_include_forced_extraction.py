import numpy as np
from aspired import image_reduction
from aspired import spectral_reduction

spatial_mask = np.arange(50, 220)
spec_mask = np.arange(50, 1024)

#
# Extract two spectra in a single science frame
#

# Science frame
lhs6328_frame = image_reduction.ImageReduction(
    log_file_name=None,
    log_level='INFO')
lhs6328_frame.add_filelist('example_use_cases/sprat_LHS6328.list')
lhs6328_frame.load_data()
lhs6328_frame.reduce()
lhs6328_frame.inspect(filename='example_output/example_01_a_science_image',
                      save_fig=True)
lhs6328_frame.save_fits(filename='example_output/example_01_a_science_image',
                        overwrite=True)

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

# Force extraction
lhs6328_twodspec.ap_extract(
    apwidth=15,
    skywidth=10,
    skydeg=1,
    optimal=True,
    forced=True,
    variances=lhs6328_twodspec.spectrum_list[1].var,
    display=True,
    filename='example_output/example_01_a_science_apextract_forced_weighted',
    save_fig=True)

# Aperture extraction
lhs6328_twodspec.ap_extract(
    apwidth=15,
    skywidth=10,
    skydeg=1,
    optimal=False,
    display=False,
    filename='example_output/example_01_a_science_apextract_tophat',
    save_fig=True)

# Optimal extraction
lhs6328_twodspec.ap_extract(
    apwidth=15,
    skywidth=10,
    skydeg=1,
    optimal=True,
    forced=True,
    variances=1000000.,
    display=False,
    filename=''
    'example_output/example_01_a_science_apextract_forced_unit_weighted',
    save_fig=True)

lhs6328_twodspec.save_fits(
    filename='example_output/example_01_a_science_traces', overwrite=True)

# Standard frame
standard_frame = image_reduction.ImageReduction(log_file_name=None)
standard_frame.add_filelist('example_use_cases/sprat_Hiltner102.list')
standard_frame.load_data()
standard_frame.reduce()
standard_frame.inspect(filename='example_output/example_01_a_standard_image',
                       save_fig=True)
standard_frame.save_fits(filename='example_output/example_01_a_standard_image',
                         overwrite=True)

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

hilt102_twodspec.save_fits(
    filename='example_output/example_01_a_standard_trace', overwrite=True)

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

lhs6328_onedspec.add_atlas(elements=['Xe'],
                           min_atlas_wavelength=4000,
                           max_atlas_wavelength=8000,
                           min_intensity=50,
                           constrain_poly=True,
                           stype='science+standard')

# Solve for the pixel-to-wavelength solution
lhs6328_onedspec.do_hough_transform(brute_force=False)
lhs6328_onedspec.fit(max_tries=1000, stype='science+standard', display=True)
lhs6328_onedspec.plot_search_space(display=True)

# Apply the wavelength calibration and display it
lhs6328_onedspec.apply_wavelength_calibration(stype='science+standard')

# Get the standard from the library
lhs6328_onedspec.load_standard(target='hiltner102')
lhs6328_onedspec.inspect_standard(
    save_fig=True, filename='example_output/example_01_a_literature_standard')

lhs6328_onedspec.compute_sensitivity(k=3, mask_fit_size=1)
lhs6328_onedspec.inspect_sensitivity(
    save_fig=True, filename='example_output/example_01_a_sensitivity')

lhs6328_onedspec.apply_flux_calibration(stype='science+standard')

# Apply atmospheric extinction correction
lhs6328_onedspec.set_atmospheric_extinction(location='orm')
lhs6328_onedspec.apply_atmospheric_extinction_correction()

# Create FITS
lhs6328_onedspec.create_fits(output='trace+count')

# Modify FITS header for the trace
lhs6328_onedspec.modify_trace_header(0, 'set', 'COMMENT', 'Hello World!')

print(lhs6328_onedspec.science_spectrum_list[0].trace_hdulist[0].header)

lhs6328_onedspec.inspect_reduced_spectrum(
    stype='science',
    save_fig=True,
    filename='example_output/example_01_a_science_spectrum')

lhs6328_onedspec.inspect_reduced_spectrum(
    stype='standard',
    save_fig=True,
    filename='example_output/example_01_a_standard_spectrum')

# Save as FITS
lhs6328_onedspec.save_fits(
    output='flux_resampled+wavecal+flux+count+trace',
    filename='example_output/example_01_a_full_reduction',
    stype='science+standard',
    overwrite=True)

# Check the modified headers are still not overwritten
print(lhs6328_onedspec.science_spectrum_list[0].trace_hdulist[0].header)

# save as CSV
lhs6328_onedspec.save_csv(
    output='flux_resampled+wavecal+flux+count+trace',
    filename='example_output/example_01_a_full_reduction',
    stype='science+standard',
    overwrite=True)
