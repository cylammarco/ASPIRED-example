from astropy.io import fits
from aspired import spectral_reduction
import numpy as np

# Load the image
lhs6328_fits = fits.open(
    'example_use_cases/sprat_LHS6328_Hiltner102_raw/v_e_20180810_12_1_0_0.fits.gz')[0]

spatial_mask = np.arange(50, 200)
spec_mask = np.arange(50, 1024)

#
# Loading a single pre-saved spectral trace.
#
lhs6328_extracted = fits.open(
    'example_output/example_01_a_full_reduction_science_0.fits')
lhs6328_trace = lhs6328_extracted[1].data
lhs6328_trace_sigma = lhs6328_extracted[2].data

lhs6328_twodspec = spectral_reduction.TwoDSpec(lhs6328_fits,
                                               spatial_mask=spatial_mask,
                                               spec_mask=spec_mask,
                                               readnoise=2.34)

# Adding the trace and the line spread function (sigma) to the TwoDSpec
# object
lhs6328_twodspec.add_trace(trace=lhs6328_trace,
                           trace_sigma=lhs6328_trace_sigma)

lhs6328_twodspec.ap_extract(apwidth=15,
                            optimal=True,
                            skywidth=10,
                            skydeg=1,
                            display=True)

lhs6328_twodspec.save_fits(
    output='count',
    filename=
    'example_output/example_04_user_supplied_trace_for_spectral_extraction',
    overwrite=True)
