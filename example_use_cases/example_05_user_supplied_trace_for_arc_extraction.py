import os

import numpy as np
from aspired import spectral_reduction
from astropy.io import fits

HERE = os.getcwd()

# Load the image
arc_fits = fits.open(
    os.path.join(
        HERE,
        "sprat_LHS6328_Hiltner102_raw",
        "v_a_20180810_13_1_0_1.fits.gz",
    )
)[0]

spatial_mask = np.arange(50, 200)
spec_mask = np.arange(50, 1024)

#
# Loading a pre-saved spectral trace.
#

lhs6328_extracted = fits.open(
    os.path.join(
        HERE,
        "..",
        "example_output",
        "example_01_a_full_reduction_science_0.fits",
    )
)
lhs6328_trace = lhs6328_extracted[1].data
lhs6328_trace_sigma = lhs6328_extracted[2].data
lhs6328_adu = lhs6328_extracted[3].data

lhs6328_twodspec = spectral_reduction.TwoDSpec(
    spatial_mask=spatial_mask, spec_mask=spec_mask, readnoise=2.34
)

# Add a 2D arc image
lhs6328_twodspec.add_arc(arc_fits)
lhs6328_twodspec.apply_mask_to_arc()

# Add the trace and the line spread function (sigma) to the 2D arc image
lhs6328_twodspec.add_trace(lhs6328_trace, lhs6328_trace_sigma)

# Extract the 1D arc by aperture sum of the traces provided
lhs6328_twodspec.extract_arc_spec(display=True)

lhs6328_onedspec = spectral_reduction.OneDSpec()
lhs6328_onedspec.from_twodspec(lhs6328_twodspec, stype="science")

# Find the peaks of the arc
lhs6328_onedspec.find_arc_lines(display=True, stype="science")

lhs6328_onedspec.save_fits(
    output="arc_spec",
    filename=os.path.join(
        HERE, "..", "example_output", "example_05_arcspec_1"
    ),
    stype="science",
    overwrite=True,
)
