from astropy.io import fits
from aspired import spectral_reduction


# Load the image
arc_fits = fits.open(
    'sprat_LHS6328_Hiltner102_raw/v_a_20180810_13_1_0_1.fits.gz')[0]

#
# Loading a pre-saved spectral trace.
#

lhs6328_extracted = fits.open(
    'example_output/example_01_a_full_reduction_science_0.fits')
lhs6328_trace = lhs6328_extracted[1].data
lhs6328_trace_sigma = lhs6328_extracted[2].data
lhs6328_adu = lhs6328_extracted[3].data

lhs6328_onedspec = spectral_reduction.OneDSpec()

# Add a 2D arc image
lhs6328_onedspec.add_arc(arc_fits, stype='science')

# Add the trace and the line spread function (sigma) to the 2D arc image
lhs6328_onedspec.add_trace(lhs6328_trace, lhs6328_trace_sigma, stype='science')

# Extract the 1D arc by aperture sum of the traces provided
lhs6328_onedspec.extract_arc_spec(display=True, stype='science')

# Find the peaks of the arc
lhs6328_onedspec.find_arc_lines(display=True, stype='science')

lhs6328_onedspec.wavecal_science.save_fits(output='arc_spec', filename='example_output/example_05_arcspec_1')

