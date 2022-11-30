import copy
import os

import numpy as np
from aspired import spectral_reduction
from astroscrappy import detect_cosmics
from astropy import units as u
from astropy.io import fits
from astropy.nddata import CCDData
from ccdproc import Combiner
import extinction
from matplotlib import pyplot as plt
from spectres import spectres

plt.ion()

lines_H = [6562.79, 4861.35, 4340.472, 4101.734, 3970.075, 3889.064, 3835.397]
lines_HeI = [10830, 7065, 6678, 5876, 5016, 4922, 4713, 4472, 4026, 3965, 3889]
lines_HeI_highres = [
    3819.607,
    3819.76,
    3888.6489,
    3964.729,
    4009.27,
    4026.191,
    4120.82,
    4143.76,
    4387.929,
    4437.55,
    4471.479,
]
lines_HeII = [5411.52, 4685.7, 4542.8, 4340, 4200, 3203.10]
lines_CII = [
    3918.978,
    3920.693,
    4267.003,
    4267.258,
    5145.16,
    5151.09,
    5889.77,
    6578.05,
    6582.88,
    7231.32,
    7236.42,
]
lines_NII = [3995.00, 4630.54, 5005.15, 5679.56, 6482.05, 6610.56]
lines_OI = [9263, 8446, 7775, 7774, 7772, 6158]
lines_OI_forbidden = [6363, 6300, 5577]
lines_OII = [6721, 6641, 4649, 4416, 4317, 4076, 3982, 3973, 3911, 3749, 3713]
lines_OII_forbidden = [3729, 3726]
lines_OIII_forbidden = [5007, 4959, 4363]
lines_NaI = [8195, 8183, 5896, 5890]
lines_MgI = [
    8807,
    5528,
    5184,
    5173,
    5167,
    4703,
    4571,
    3838,
    3832,
    3829,
    2852,
    2780,
]
lines_MgII = [9632, 9244, 9218, 8235, 8214, 7896, 7877, 4481, 2803, 2796, 2791]
lines_SiII = [7850, 6371, 6347, 5979, 5958, 5670, 5056, 5041, 4131, 4128, 3856]
lines_SII = [6715, 6397, 6313, 6305, 6287, 5647, 5640, 5606, 5454, 5433, 4163]
lines_CaII = [8662, 8542, 8498, 3969, 3934, 3737, 3706, 3180, 3159]
lines_CaII_forbidden = [7324, 7292]
lines_FeII = [5363, 5235, 5198, 5169, 5018, 4924, 4549, 4515, 4352, 4303]
lines_FeIII = [5158, 5129, 4432, 4421, 4397]

lines_OI_atm = [5577, 6300.3, 7774.2]
lines_OH_atm = [6863, 7340, 7523, 7662, 7750, 7821, 7913, 8025]
lines_O2_atm = [6277.7, 7605, 6869]
lines_H2O_atm = [
    6940.7,
    7185.9,
    7234.1,
    7253.2,
    7276.2,
    7292.0,
    7303.2,
    7318.4,
]

base_folder = "OB0001"

bias_1 = "bias/0002673108-20200913-OSIRIS-OsirisBias.fits"
bias_2 = "bias/0002673110-20200913-OSIRIS-OsirisBias.fits"
bias_3 = "bias/0002673111-20200913-OSIRIS-OsirisBias.fits"
bias_4 = "bias/0002673112-20200913-OSIRIS-OsirisBias.fits"
bias_5 = "bias/0002673114-20200913-OSIRIS-OsirisBias.fits"
bias_6 = "bias/0002673115-20200913-OSIRIS-OsirisBias.fits"
bias_7 = "bias/0002673116-20200913-OSIRIS-OsirisBias.fits"
bias_8 = "bias/0002673117-20200913-OSIRIS-OsirisBias.fits"
bias_9 = "bias/0002673118-20200913-OSIRIS-OsirisBias.fits"
bias_10 = "bias/0002673119-20200913-OSIRIS-OsirisBias.fits"
bias_11 = "bias/0002673120-20200913-OSIRIS-OsirisBias.fits"

light_r1000b_1 = (
    "object/0002673185-20200913-OSIRIS-OsirisLongSlitSpectroscopy.fits"
)
light_r1000b_2 = (
    "object/0002673186-20200913-OSIRIS-OsirisLongSlitSpectroscopy.fits"
)

standard_r1000b = (
    "stds/0002673244-20200913-OSIRIS-OsirisLongSlitSpectroscopy.fits"
)

flat_r1000b_1 = "flat/0002672527-20200911-OSIRIS-OsirisSpectralFlat.fits"
flat_r1000b_2 = "flat/0002672528-20200911-OSIRIS-OsirisSpectralFlat.fits"
flat_r1000b_3 = "flat/0002672530-20200911-OSIRIS-OsirisSpectralFlat.fits"
flat_r1000b_4 = "flat/0002672533-20200911-OSIRIS-OsirisSpectralFlat.fits"

hgar_r1000b = "arc/0002672524-20200911-OSIRIS-OsirisCalibrationLamp.fits"
ne_r1000b_1 = "arc/0002672525-20200911-OSIRIS-OsirisCalibrationLamp.fits"
ne_r1000b_2 = "arc/0002672526-20200911-OSIRIS-OsirisCalibrationLamp.fits"

# Creates a master bias frame using median combine
bias_CCDData = []
bias_frames = [
    bias_1,
    bias_2,
    bias_3,
    bias_4,
    bias_5,
    bias_6,
    bias_7,
    bias_8,
    bias_9,
    bias_10,
    bias_11,
]

for i in bias_frames:
    # Open all the bias frames
    bias = fits.open(os.path.join(base_folder, i))[2]
    bias_CCDData.append(CCDData(bias.data, unit=u.ct))

# Put data into a Combiner
bias_combiner = Combiner(bias_CCDData)

# Apply sigma clipping
bias_combiner.sigma_clipping(
    low_thresh=5.0, high_thresh=5.0, func=np.ma.median
)

bias_master = bias_combiner.median_combine()

# Free memory
bias = None
bias_CCDData = None
bias_combiner = None

#
#
# Process the R1000B here
#
#

# Creates a master flat frame using median combine
flat_r1000b_CCDData = []
flat_r1000b_frames = [
    flat_r1000b_1,
    flat_r1000b_2,
    flat_r1000b_3,
    flat_r1000b_4,
]

for i in flat_r1000b_frames:
    # Open all the flat frames
    flat_r1000b = fits.open(os.path.join(base_folder, i))[2]
    flat_r1000b_CCDData.append(CCDData(flat_r1000b.data, unit=u.ct))

# Put data into a Combiner
flat_r1000b_combiner = Combiner(flat_r1000b_CCDData)

# Apply sigma clipping
flat_r1000b_combiner.sigma_clipping(
    low_thresh=5.0, high_thresh=5.0, func=np.ma.median
)

flat_r1000b_master = flat_r1000b_combiner.median_combine()

# Free memory
flat_r1000b = None
flat_r1000b_CCDData = None
flat_r1000b_combiner = None

# sum the arcs
arc_r1000b_CCDData = []
arc_r1000b_frames = [hgar_r1000b, ne_r1000b_1, ne_r1000b_2]

for i in arc_r1000b_frames:
    # Open all the arc frames
    arc_r1000b = fits.open(os.path.join(base_folder, i))[2]
    arc_r1000b_CCDData.append(CCDData(arc_r1000b.data, unit=u.ct))

# Put data into a Combiner
arc_r1000b_combiner = Combiner(arc_r1000b_CCDData)

# Apply sigma clipping
arc_r1000b_combiner.sigma_clipping(
    low_thresh=5.0, high_thresh=5.0, func=np.ma.median
)

arc_r1000b_master = arc_r1000b_combiner.sum_combine()

# Free memory
arc_r1000b = None
arc_r1000b_CCDData = None
arc_r1000b_combiner = None

spatial_mask = np.arange(125, 325)

light_fits_1 = fits.open(os.path.join(base_folder, light_r1000b_1))
light_fits_2 = fits.open(os.path.join(base_folder, light_r1000b_2))
standard_fits = fits.open(os.path.join(base_folder, standard_r1000b))

flat_master = flat_r1000b_master.data - bias_master.data
flat_normalised = flat_master / np.nanmean(flat_master)

light_frame_1 = light_fits_1[2].data.astype("float")
light_frame_1 = detect_cosmics(
    light_frame_1 / 0.95, gain=0.95, readnoise=4.5, sigclip=3.0
)[1]
light_frame_1 -= bias_master.data
light_frame_1 /= flat_normalised

light_frame_2 = light_fits_2[2].data.astype("float")
light_frame_2 = detect_cosmics(
    light_frame_2 / 0.95, gain=0.95, readnoise=4.5, sigclip=3.0
)[1]
light_frame_2 -= bias_master.data
light_frame_2 /= flat_normalised

standard_frame = standard_fits[2].data.astype("float")
standard_frame = detect_cosmics(
    standard_frame / 0.95, gain=0.95, readnoise=4.5, sigclip=3.0
)[1]
standard_frame -= bias_master.data
standard_frame /= flat_normalised

science_1_twodspec = spectral_reduction.TwoDSpec(
    light_frame_1, light_fits_1[0].header, spatial_mask=spatial_mask, saxis=0
)

science_2_twodspec = spectral_reduction.TwoDSpec(
    light_frame_2, light_fits_2[0].header, spatial_mask=spatial_mask, saxis=0
)

standard_twodspec = spectral_reduction.TwoDSpec(
    standard_frame, standard_fits[0].header, spatial_mask=spatial_mask, saxis=0
)

science_1_twodspec.add_arc(arc_r1000b_master.data)
science_1_twodspec.apply_mask_to_arc()
science_1_twodspec.ap_trace(
    display=False,
    fit_deg=3,
    save_fig=True,
    fig_type="jpg",
    filename="R1000B_1_aptrace",
)
science_1_twodspec.ap_extract(
    display=False,
    apwidth=7,
    skywidth=5,
    save_fig=True,
    fig_type="jpg",
    filename="R1000B_1_apextract",
)
science_1_twodspec.extract_arc_spec(spec_width=20, display=False)

science_2_twodspec.add_arc(arc_r1000b_master.data)
science_2_twodspec.apply_mask_to_arc()
science_2_twodspec.ap_trace(
    display=False,
    fit_deg=2,
    save_fig=True,
    fig_type="jpg",
    filename="R1000B_2_aptrace",
)
science_2_twodspec.ap_extract(
    display=False,
    apwidth=7,
    skywidth=5,
    save_fig=True,
    fig_type="jpg",
    filename="R1000B_2_apextract",
)
science_2_twodspec.extract_arc_spec(spec_width=20, display=False)

standard_twodspec.add_arc(arc_r1000b_master.data)
standard_twodspec.apply_mask_to_arc()
standard_twodspec.ap_trace(display=False, fit_deg=2)
standard_twodspec.ap_extract(display=False, apwidth=15, skywidth=9)
standard_twodspec.extract_arc_spec(spec_width=20, display=False)

# One dimensional spectral operation
science_1_onedspec = spectral_reduction.OneDSpec(
    log_file_name=None, log_level="INFO"
)
science_1_onedspec.from_twodspec(science_1_twodspec, stype="science")
science_1_onedspec.from_twodspec(
    copy.copy(standard_twodspec), stype="standard"
)

science_2_onedspec = spectral_reduction.OneDSpec(
    log_file_name=None, log_level="INFO"
)
science_2_onedspec.from_twodspec(science_2_twodspec, stype="science")
science_2_onedspec.from_twodspec(
    copy.copy(standard_twodspec), stype="standard"
)

science_1_onedspec.find_arc_lines(
    prominence=0.0001,
    distance=3,
    top_n_peaks=50,
    refine_window_width=3,
    display=False,
)
science_2_onedspec.find_arc_lines(
    prominence=0.0001,
    distance=3,
    top_n_peaks=50,
    refine_window_width=3,
    display=False,
)

atlas = [
    3650.153,
    4046.563,
    4358.328,
    5460.735,
    5769.598,
    5790.663,
    6682.960,
    6752.834,
    6871.289,
    6965.431,
    7030.251,
    7067.218,
    7147.042,
    7272.936,
    7383.981,
    7503.869,
    7514.652,
    7635.106,
    7723.98,
    5852.488,
    5881.895,
    5944.834,
    5975.534,
    6074.338,
    6096.163,
    6266.495,
    6334.428,
    6402.248,
    6506.528,
    6532.822,
    6598.953,
    6678.276,
    6717.043,
    6929.467,
    7032.413,
    7173.938,
    7245.167,
    7438.899,
    7488.871,
    7535.774,
]
element = ["HgArNe"] * len(atlas)

# Configure the wavelength calibrator
science_1_onedspec.initialise_calibrator(stype="science+standard")

science_1_onedspec.set_hough_properties(
    num_slopes=5000,
    xbins=500,
    ybins=500,
    min_wavelength=3500,
    max_wavelength=8000,
    stype="science+standard",
)
science_1_onedspec.set_ransac_properties(
    sample_size=5, minimum_matches=34, stype="science+standard"
)

science_1_onedspec.add_user_atlas(
    elements=element, wavelengths=atlas, stype="science+standard"
)

science_1_onedspec.do_hough_transform()

# Solve for the pixel-to-wavelength solution
science_1_onedspec.fit(
    max_tries=2000, fit_deg=4, stype="science+standard", display=True
)

# Apply the wavelength calibration and display it
science_1_onedspec.apply_wavelength_calibration(stype="science+standard")

# Get the standard from the library
science_1_onedspec.load_standard(target="Feige110", library="esoxshooter")
science_1_onedspec.get_continuum(lowess_frac=0.2)
science_1_onedspec.get_sensitivity(
    k=3, method="interpolate", mask_fit_size=1, lowess_frac=0.025
)
science_1_onedspec.inspect_sensitivity()

science_1_onedspec.apply_flux_calibration(stype="science+standard")

# Apply atmospheric extinction correction
science_1_onedspec.set_atmospheric_extinction(location="orm")
science_1_onedspec.apply_atmospheric_extinction_correction()

science_1_onedspec.inspect_reduced_spectrum(
    save_fig=True, filename="science_1", fig_type="png", stype="science"
)
science_1_onedspec.inspect_reduced_spectrum(
    save_fig=True, filename="standard_1", fig_type="png", stype="standard"
)

# Configure the wavelength calibrator
science_2_onedspec.initialise_calibrator(stype="science+standard")

science_2_onedspec.set_hough_properties(
    num_slopes=5000,
    xbins=500,
    ybins=500,
    min_wavelength=3500,
    max_wavelength=8000,
    stype="science+standard",
)
science_2_onedspec.set_ransac_properties(
    sample_size=5, minimum_matches=34, stype="science+standard"
)

science_2_onedspec.add_user_atlas(
    elements=element, wavelengths=atlas, stype="science+standard"
)

science_2_onedspec.do_hough_transform()

# Solve for the pixel-to-wavelength solution
science_2_onedspec.fit(
    max_tries=2000, fit_deg=4, stype="science+standard", display=True
)

# Apply the wavelength calibration and display it
science_2_onedspec.apply_wavelength_calibration(stype="science+standard")

# Get the standard from the library
science_2_onedspec.load_standard(target="Feige110", library="esoxshooter")
science_2_onedspec.get_continuum(lowess_frac=0.2)

science_2_onedspec.get_sensitivity(
    k=3, method="interpolate", mask_fit_size=1, lowess_frac=0.025
)
science_2_onedspec.inspect_sensitivity()

science_2_onedspec.apply_flux_calibration(stype="science+standard")

# Apply atmospheric extinction correction
science_2_onedspec.set_atmospheric_extinction(location="orm")
science_2_onedspec.apply_atmospheric_extinction_correction()

science_2_onedspec.inspect_reduced_spectrum(
    save_fig=True, filename="science_2", fig_type="png", stype="science"
)
science_2_onedspec.inspect_reduced_spectrum(
    save_fig=True, filename="standard_2", fig_type="png", stype="standard"
)

#science_1_onedspec.resample()
#science_2_onedspec.resample()

wave_1 = science_1_onedspec.science_spectrum_list[0].wave_resampled
wave_1_bin2 = wave_1[::2]
wave_1_bin5 = wave_1[::5]
wave_1_bin10 = wave_1[::10]

flux_1 = science_1_onedspec.science_spectrum_list[
    0
].flux_resampled

""" for v0.5
flux_1 = science_1_onedspec.science_spectrum_list[
    0
].flux_resampled_atm_ext_corrected
"""

flux_1_bin1 = flux_1
flux_1_bin2 = spectres(wave_1_bin2, wave_1, flux_1)
flux_1_bin5 = spectres(wave_1_bin5, wave_1, flux_1)
flux_1_bin10 = spectres(wave_1_bin10, wave_1, flux_1)

wave_2 = science_2_onedspec.science_spectrum_list[0].wave_resampled
flux_2 = science_2_onedspec.science_spectrum_list[
    0
].flux_resampled

""" for v0.5
flux_2 = science_2_onedspec.science_spectrum_list[
    0
].flux_resampled_atm_ext_corrected
"""
# note this is resampled to match wave_1s
flux_2_bin1 = spectres(wave_1, wave_2, flux_2)
flux_2_bin2 = spectres(wave_1_bin2, wave_2, flux_2)
flux_2_bin5 = spectres(wave_1_bin5, wave_2, flux_2)
flux_2_bin10 = spectres(wave_1_bin10, wave_2, flux_2)

# Get the interstellar dust extinction correction
ext_bin1 = extinction.fm07(wave_1, 0.81 * 3.1)
ext_bin2 = extinction.fm07(wave_1_bin2, 0.81 * 3.1)
ext_bin5 = extinction.fm07(wave_1_bin5, 0.81 * 3.1)

#
#
# Process the R2500U here
#
#

light_r2500u_1 = (
    "object/0002673193-20200913-OSIRIS-OsirisLongSlitSpectroscopy.fits"
)
light_r2500u_2 = (
    "object/0002673194-20200913-OSIRIS-OsirisLongSlitSpectroscopy.fits"
)

standard_r2500u = (
    "stds/0002673245-20200913-OSIRIS-OsirisLongSlitSpectroscopy.fits"
)

flat_r2500u_1 = "flat/0002672538-20200911-OSIRIS-OsirisSpectralFlat.fits"
flat_r2500u_2 = "flat/0002672539-20200911-OSIRIS-OsirisSpectralFlat.fits"
flat_r2500u_3 = "flat/0002672540-20200911-OSIRIS-OsirisSpectralFlat.fits"
flat_r2500u_4 = "flat/0002672541-20200911-OSIRIS-OsirisSpectralFlat.fits"

hgar_r2500u = "arc/0002672534-20200911-OSIRIS-OsirisCalibrationLamp.fits"
xe_r2500u = "arc/0002672535-20200911-OSIRIS-OsirisCalibrationLamp.fits"
xe_r2500u = "arc/0002672536-20200911-OSIRIS-OsirisCalibrationLamp.fits"

# Creates a master flat frame using median combine
flat_r2500u_CCDData = []
flat_r2500u_frames = [
    flat_r2500u_1,
    flat_r2500u_2,
    flat_r2500u_3,
    flat_r2500u_4,
]

for i in flat_r2500u_frames:
    # Open all the flat frames
    flat_r2500u = fits.open(os.path.join(base_folder, i))[2]
    flat_r2500u_CCDData.append(CCDData(flat_r2500u.data, unit=u.ct))

# Put data into a Combiner
flat_r2500u_combiner = Combiner(flat_r2500u_CCDData)

# Apply sigma clipping
flat_r2500u_combiner.sigma_clipping(
    low_thresh=5.0, high_thresh=5.0, func=np.ma.median
)

flat_r2500u_master = flat_r2500u_combiner.median_combine()

# Free memory
flat_r2500u = None
flat_r2500u_CCDData = None
flat_r2500u_combiner = None

# sum the arcs
arc_r2500u_CCDData = []
arc_r2500u_frames = [hgar_r2500u, xe_r2500u, xe_r2500u]

for i in arc_r2500u_frames:
    # Open all the arc frames
    arc_r2500u = fits.open(os.path.join(base_folder, i))[2]
    arc_r2500u_CCDData.append(CCDData(arc_r2500u.data, unit=u.ct))

# Put data into a Combiner
arc_r2500u_combiner = Combiner(arc_r2500u_CCDData)

# Apply sigma clipping
arc_r2500u_combiner.sigma_clipping(
    low_thresh=5.0, high_thresh=5.0, func=np.ma.median
)

arc_r2500u_master = arc_r2500u_combiner.sum_combine()

# Free memory
arc_r2500u = None
arc_r2500u_CCDData = None
arc_r2500u_combiner = None

spatial_mask_r2500u = np.arange(125, 325)

light_fits_r2500u_1 = fits.open(os.path.join(base_folder, light_r2500u_1))
light_fits_r2500u_2 = fits.open(os.path.join(base_folder, light_r2500u_2))
standard_fits_r2500u = fits.open(os.path.join(base_folder, standard_r2500u))

flat_r2500u_master = flat_r2500u_master.data - bias_master.data
flat_r2500u_normalised = flat_r2500u_master / np.nanmean(flat_r2500u_master)

light_frame_r2500u_1 = light_fits_r2500u_1[2].data.astype("float")
light_frame_r2500u_1 = detect_cosmics(
    light_frame_r2500u_1 / 0.95, gain=0.95, readnoise=4.5, sigclip=3.0
)[1]
light_frame_r2500u_1 -= bias_master.data
light_frame_r2500u_1 /= flat_r2500u_normalised

light_frame_r2500u_2 = light_fits_r2500u_2[2].data.astype("float")
light_frame_r2500u_2 = detect_cosmics(
    light_frame_r2500u_2 / 0.95, gain=0.95, readnoise=4.5, sigclip=3.0
)[1]
light_frame_r2500u_2 -= bias_master.data
light_frame_r2500u_2 /= flat_r2500u_normalised

standard_frame_r2500u = standard_fits_r2500u[2].data.astype("float")
standard_frame_r2500u -= bias_master.data
standard_frame_r2500u /= flat_r2500u_normalised

science_r2500u_1_twodspec = spectral_reduction.TwoDSpec(
    light_frame_r2500u_1,
    light_fits_r2500u_1[0].header,
    spatial_mask=spatial_mask_r2500u,
    saxis=0,
)

science_r2500u_2_twodspec = spectral_reduction.TwoDSpec(
    light_frame_r2500u_2,
    light_fits_r2500u_2[0].header,
    spatial_mask=spatial_mask_r2500u,
    saxis=0,
)

standard_r2500u_twodspec = spectral_reduction.TwoDSpec(
    standard_frame_r2500u,
    standard_fits_r2500u[0].header,
    spatial_mask=spatial_mask_r2500u,
    saxis=0,
)

science_r2500u_1_twodspec.add_arc(arc_r2500u_master.data)
science_r2500u_1_twodspec.apply_mask_to_arc()
science_r2500u_1_twodspec.ap_trace(
    display=False,
    fit_deg=3,
    ap_faint=0,
    save_fig=True,
    fig_type="jpg",
    filename="R2500U_1_aptrace",
)
science_r2500u_1_twodspec.ap_extract(
    display=True,
    apwidth=11,
    skywidth=11,
    save_fig=True,
    fig_type="jpg",
    filename="R2500U_1_apextract",
)
science_r2500u_1_twodspec.extract_arc_spec(display=False)

science_r2500u_2_twodspec.add_arc(arc_r2500u_master.data)
science_r2500u_2_twodspec.apply_mask_to_arc()
science_r2500u_2_twodspec.ap_trace(
    display=False,
    fit_deg=3,
    ap_faint=0,
    save_fig=True,
    fig_type="jpg",
    filename="R2500U_2_aptrace",
)
science_r2500u_2_twodspec.ap_extract(
    display=True,
    apwidth=11,
    skywidth=11,
    save_fig=True,
    fig_type="jpg",
    filename="R2500U_2_apextract",
)
science_r2500u_2_twodspec.extract_arc_spec(display=False)

# saturated
standard_r2500u_twodspec.add_arc(arc_r2500u_master.data)
standard_r2500u_twodspec.apply_mask_to_arc()
standard_r2500u_twodspec.ap_trace(display=False, fit_deg=3, ap_faint=0)
standard_r2500u_twodspec.ap_extract(display=False, apwidth=25, skywidth=10)
standard_r2500u_twodspec.extract_arc_spec(display=False)

# One dimensional spectral operation
science_r2500u_1_onedspec = spectral_reduction.OneDSpec(
    log_file_name=None, log_level="INFO"
)
science_r2500u_1_onedspec.from_twodspec(
    science_r2500u_1_twodspec, stype="science"
)
science_r2500u_1_onedspec.from_twodspec(
    copy.copy(standard_r2500u_twodspec), stype="standard"
)

science_r2500u_2_onedspec = spectral_reduction.OneDSpec(
    log_file_name=None, log_level="INFO"
)
science_r2500u_2_onedspec.from_twodspec(
    science_r2500u_2_twodspec, stype="science"
)
science_r2500u_2_onedspec.from_twodspec(
    copy.copy(standard_r2500u_twodspec), stype="standard"
)

science_r2500u_1_onedspec.find_arc_lines(
    prominence=0.01,
    distance=3,
    refine_window_width=3,
    top_n_peaks=15,
    display=False,
)
science_r2500u_2_onedspec.find_arc_lines(
    prominence=0.01,
    distance=3,
    refine_window_width=3,
    top_n_peaks=15,
    display=False,
)

atlas_r2500u = [
    3650.153,
    3948.979,
    4046.563,
    4077.831,
    4191.029,
    4358.328,
    4500.977,
    4524.680,
    4582.747,
    4624.276,
    4671.226,
    4697.02,
]
element_r2500u = ["HgArXe"] * len(atlas_r2500u)

# Configure the wavelength calibrator
science_r2500u_1_onedspec.initialise_calibrator(stype="science+standard")

science_r2500u_1_onedspec.set_hough_properties(
    num_slopes=2000,
    xbins=200,
    ybins=200,
    min_wavelength=3000,
    max_wavelength=4800,
    stype="science+standard",
)
science_r2500u_1_onedspec.set_ransac_properties(
    filter_close=True,
    sample_size=5,
    minimum_matches=8,
    stype="science+standard",
)

science_r2500u_1_onedspec.add_user_atlas(
    elements=element_r2500u, wavelengths=atlas_r2500u, stype="science+standard"
)

science_r2500u_1_onedspec.do_hough_transform()

# Solve for the pixel-to-wavelength solution
science_r2500u_1_onedspec.fit(
    max_tries=1000, stype="science+standard", display=True
)

# Apply the wavelength calibration and display it
science_r2500u_1_onedspec.apply_wavelength_calibration(
    stype="science+standard"
)

# Get the standard from the library
science_r2500u_1_onedspec.load_standard(
    target="Feige110", library="esoxshooter"
)
science_r2500u_1_onedspec.get_continuum(lowess_frac=0.5)
science_r2500u_1_onedspec.get_sensitivity(
    k=3, method="interpolate", mask_fit_size=1, lowess_frac=0.025
)
science_r2500u_1_onedspec.inspect_sensitivity()

science_r2500u_1_onedspec.apply_flux_calibration(stype="science+standard")

# Apply atmospheric extinction correction
science_r2500u_1_onedspec.set_atmospheric_extinction(location="orm")
science_r2500u_1_onedspec.apply_atmospheric_extinction_correction()

science_r2500u_1_onedspec.inspect_reduced_spectrum(
    wave_min=3000.0, wave_max=5000.0
)

# Configure the wavelength calibrator
science_r2500u_2_onedspec.initialise_calibrator(stype="science+standard")

science_r2500u_2_onedspec.set_hough_properties(
    num_slopes=2000,
    xbins=200,
    ybins=200,
    min_wavelength=3000,
    max_wavelength=4800,
    stype="science+standard",
)
science_r2500u_2_onedspec.set_ransac_properties(
    filter_close=True,
    sample_size=5,
    minimum_matches=8,
    stype="science+standard",
)

science_r2500u_2_onedspec.add_user_atlas(
    elements=element_r2500u, wavelengths=atlas_r2500u, stype="science+standard"
)

science_r2500u_2_onedspec.do_hough_transform()

# Solve for the pixel-to-wavelength solution
science_r2500u_2_onedspec.fit(
    max_tries=1000, stype="science+standard", display=True
)

# Apply the wavelength calibration and display it
science_r2500u_2_onedspec.apply_wavelength_calibration(
    stype="science+standard"
)

# Get the standard from the library
science_r2500u_2_onedspec.load_standard(
    target="Feige110", library="esoxshooter"
)
science_r2500u_2_onedspec.get_continuum(lowess_frac=0.5)
science_r2500u_2_onedspec.get_sensitivity(
    k=3, method="interpolate", mask_fit_size=1, lowess_frac=0.025
)
science_r2500u_2_onedspec.inspect_sensitivity()

science_r2500u_2_onedspec.apply_flux_calibration(stype="science+standard")

# Apply atmospheric extinction correction
science_r2500u_2_onedspec.set_atmospheric_extinction(location="orm")
science_r2500u_2_onedspec.apply_atmospheric_extinction_correction()

science_r2500u_2_onedspec.inspect_reduced_spectrum(
    wave_min=3000.0, wave_max=5000.0
)


#science_r2500u_1_onedspec.resample()
#science_r2500u_2_onedspec.resample()

wave_r2500u_1 = science_r2500u_1_onedspec.science_spectrum_list[
    0
].wave_resampled
wave_r2500u_1_bin2 = wave_r2500u_1[::2]
wave_r2500u_1_bin5 = wave_r2500u_1[::5]
wave_r2500u_1_bin10 = wave_r2500u_1[::10]

flux_r2500u_1 = science_r2500u_1_onedspec.science_spectrum_list[
    0
].flux_resampled
""" for v0.5
flux_r2500u_1 = science_r2500u_1_onedspec.science_spectrum_list[
    0
].flux_resampled_atm_ext_corrected
"""

flux_r2500u_1_bin1 = flux_r2500u_1
flux_r2500u_1_bin2 = spectres(wave_r2500u_1_bin2, wave_r2500u_1, flux_r2500u_1)
flux_r2500u_1_bin5 = spectres(wave_r2500u_1_bin5, wave_r2500u_1, flux_r2500u_1)
flux_r2500u_1_bin10 = spectres(
    wave_r2500u_1_bin10, wave_r2500u_1, flux_r2500u_1
)

wave_r2500u_2 = science_r2500u_2_onedspec.science_spectrum_list[
    0
].wave_resampled

flux_r2500u_2 = science_r2500u_2_onedspec.science_spectrum_list[
    0
].flux_resampled
""" for v0.5
flux_r2500u_2 = science_r2500u_2_onedspec.science_spectrum_list[
    0
].flux_resampled_atm_ext_corrected
"""

# note this is resampled to match wave_1s
flux_r2500u_2_bin1 = spectres(wave_r2500u_1, wave_r2500u_2, flux_r2500u_2)
flux_r2500u_2_bin2 = spectres(wave_r2500u_1_bin2, wave_r2500u_2, flux_r2500u_2)
flux_r2500u_2_bin5 = spectres(wave_r2500u_1_bin5, wave_r2500u_2, flux_r2500u_2)
flux_r2500u_2_bin10 = spectres(
    wave_r2500u_1_bin10, wave_r2500u_2, flux_r2500u_2
)

# Get the interstellar dust extinction correction
ext_r2500u_bin1 = extinction.fm07(wave_r2500u_1, 0.81 * 3.1)
ext_r2500u_bin2 = extinction.fm07(wave_r2500u_1_bin2, 0.81 * 3.1)
ext_r2500u_bin5 = extinction.fm07(wave_r2500u_1_bin5, 0.81 * 3.1)

ymax = np.nanmax(extinction.remove(ext_r2500u_bin1, flux_r2500u_1_bin1))

fig, (ax1, ax2) = plt.subplots(2, figsize=(15, 9))

ax1.plot(
    wave_r2500u_1,
    extinction.remove(
        ext_r2500u_bin1,
        np.nanmean((flux_r2500u_1_bin1, flux_r2500u_2_bin1), axis=0),
    ),
    color="0.0",
    label="R2500U average",
)
ax1.plot(
    wave_r2500u_1,
    extinction.remove(ext_r2500u_bin1, flux_r2500u_1_bin1),
    lw=0.5,
    label="Epoch 1",
)
ax1.plot(
    wave_r2500u_1,
    extinction.remove(ext_r2500u_bin1, flux_r2500u_2_bin1),
    lw=0.5,
    label="Epoch 2",
)
ax1.vlines(lines_H, 0, ymax, color="C0", label="H")
ax1.vlines(lines_HeI_highres, 0, ymax, color="C1", label="He I")
ax1.vlines(lines_HeII, 0, ymax, color="C2", label="He II")
ax1.vlines(lines_CII, 0, ymax, color="C3", label="C II")
ax1.vlines(lines_NII, 0, ymax, color="C4", label="N II")
ax1.vlines(lines_OII, 0, ymax, color="C5", label="O II")
ax1.vlines(lines_MgII, 0, ymax, color="C6", label="Mg II")
# C7 is grey, reserved for atmosphere...
ax1.vlines(lines_SiII, 0, ymax, color="C8", label="Si II")
ax1.vlines(lines_CaII, 0, ymax, color="C9", label="Ca II")

ax1.set_xlim(3600, 4600)
ax1.set_ylim(1.0e-14, 6.0e-14)
ax1.set_xlabel("Wavelength / A")
ax1.set_ylabel("Flux / ( erg / cm / cm / s / A)")
ax1.grid()
ax1.legend(loc="upper right", ncol=2)

ax2.plot(
    wave_1,
    extinction.remove(ext_bin1, np.nanmean((flux_1_bin1, flux_2_bin1), axis=0)),
    color="0.2",
    label="R1000B average",
)
ax2.plot(
    wave_1,
    extinction.remove(ext_bin1, flux_1_bin1),
    lw=0.5,
    label="Epoch 1",
)
ax2.plot(
    wave_1,
    extinction.remove(ext_bin1, flux_2_bin1),
    lw=0.5,
    label="Epoch 2",
)
ax2.vlines(lines_H, 0, ymax, color="C0", label="H")
ax2.vlines(lines_HeI, 0, ymax, color="C1", label="He I")
ax2.vlines(lines_HeI_highres, 0, ymax, color="C1")
ax2.vlines(lines_HeII, 0, ymax, color="C2", label="He II")
ax2.vlines(lines_CII, 0, ymax, color="C3", label="C II")
ax2.vlines(lines_NII, 0, ymax, color="C4", label="N II")
ax2.vlines(lines_OII, 0, ymax, color="C5", label="O II")
ax2.vlines(lines_MgII, 0, ymax, color="C6", label="Mg II")
# C7 is grey, reserved for atmosphere...
ax2.vlines(lines_SiII, 0, ymax, color="C8", label="Si II")

ax2.vlines(
    lines_OI_atm, 0, ymax, color="grey", alpha=0.25, lw=8, label="Atmosphere"
)
ax2.vlines(lines_O2_atm, 0, ymax, color="grey", alpha=0.25, lw=8)
ax2.vlines(lines_OH_atm, 0, ymax, color="grey", alpha=0.25, lw=8)
ax2.vlines(lines_H2O_atm, 0, ymax, color="grey", alpha=0.25, lw=8)

ax2.set_xlim(3650, 7800)
ax2.set_ylim(0.0, 0.75e-13)
ax2.set_xlabel("Wavelength / A")
ax2.set_ylabel("Flux / ( erg / cm / cm / s / A)")
ax2.grid()
ax2.legend(ncol=2)

plt.subplots_adjust(left=0.05, bottom=0.06, right=0.98, top=0.98, hspace=0.15)

plt.savefig("ZTF_BLAP_01_GTC.png")
plt.savefig("ZTF_BLAP_01_GTC.pdf")

np.save(
    "r1000b_1.npy",
    np.column_stack((wave_1, extinction.remove(ext_bin1, flux_1_bin1))),
)
np.save(
    "r1000b_2.npy",
    np.column_stack((wave_1, extinction.remove(ext_bin1, flux_2_bin1))),
)

np.save(
    "r2000u_1.npy",
    np.column_stack(
        (wave_r2500u_1, extinction.remove(ext_r2500u_bin1, flux_r2500u_1_bin1))
    ),
)
np.save(
    "r2000u_2.npy",
    np.column_stack(
        (wave_r2500u_1, extinction.remove(ext_r2500u_bin1, flux_r2500u_2_bin1))
    ),
)
