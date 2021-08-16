import os

import extinction
import numpy as np
from aspired import spectral_reduction
from astroscrappy import detect_cosmics
from astropy import units as u
from astropy.io import fits
from astropy.nddata import CCDData
from ccdproc import Combiner
from matplotlib import pyplot as plt
from scipy import signal
from spectres import spectres

lines_H = [6562.79, 4861.35, 4340.472, 4101.734, 3970.075, 3889.064, 3835.397]
lines_HeI = [10830, 7065, 6678, 5876, 5016, 4922, 4713, 4472, 4026, 3965, 3889]
lines_HeII = [5412, 4686, 4542, 3203]
lines_OI = [9263, 8446, 7775, 7774, 7772, 6158]
lines_OI_forbidden = [6363, 6300, 5577]
lines_OII = [6721, 6641, 4649, 4416, 4076, 3973, 3749, 3713]
lines_OII_forbidden = [3729, 3726]
lines_OIII_forbidden = [5007, 4959, 4363]
lines_NaI = [8195, 8183, 5896, 5890]
lines_MgI = [
    8807, 5528, 5184, 5173, 5167, 4703, 4571, 3838, 3832, 3829, 2852, 2780
]
lines_MgII = [9632, 9244, 9218, 8235, 8214, 7896, 7877, 4481, 2803, 2796, 2791]
lines_SiII = [7850, 6371, 6347, 5979, 5958, 5670, 5056, 5041, 4131, 4128, 3856]
lines_SII = [6715, 6397, 6313, 6305, 6287, 5647, 5640, 5606, 5454, 5433, 4163]
lines_CaII = [8662, 8542, 8498, 3969, 3934, 3737, 3706, 3180, 3159]
lines_CaII_forbidden = [7324, 7292]
lines_FeII = [5363, 5235, 5198, 5169, 5018, 4924, 4549, 4515, 4352, 4303]
lines_FeIII = [5158, 5129, 4432, 4421, 4397]

base_folder = 'OB0001'

bias_1 = 'bias/0002673108-20200913-OSIRIS-OsirisBias.fits'
bias_2 = 'bias/0002673110-20200913-OSIRIS-OsirisBias.fits'
bias_3 = 'bias/0002673111-20200913-OSIRIS-OsirisBias.fits'
bias_4 = 'bias/0002673112-20200913-OSIRIS-OsirisBias.fits'
bias_5 = 'bias/0002673114-20200913-OSIRIS-OsirisBias.fits'
bias_6 = 'bias/0002673115-20200913-OSIRIS-OsirisBias.fits'
bias_7 = 'bias/0002673116-20200913-OSIRIS-OsirisBias.fits'
bias_8 = 'bias/0002673117-20200913-OSIRIS-OsirisBias.fits'
bias_9 = 'bias/0002673118-20200913-OSIRIS-OsirisBias.fits'
bias_10 = 'bias/0002673119-20200913-OSIRIS-OsirisBias.fits'
bias_11 = 'bias/0002673120-20200913-OSIRIS-OsirisBias.fits'

light_r1000b_1 =\
    'object/0002673185-20200913-OSIRIS-OsirisLongSlitSpectroscopy.fits'
light_r1000b_2 =\
    'object/0002673186-20200913-OSIRIS-OsirisLongSlitSpectroscopy.fits'

standard_r1000b =\
    'stds/0002673244-20200913-OSIRIS-OsirisLongSlitSpectroscopy.fits'

flat_r1000b_1 = 'flat/0002672527-20200911-OSIRIS-OsirisSpectralFlat.fits'
flat_r1000b_2 = 'flat/0002672528-20200911-OSIRIS-OsirisSpectralFlat.fits'
flat_r1000b_3 = 'flat/0002672530-20200911-OSIRIS-OsirisSpectralFlat.fits'
flat_r1000b_4 = 'flat/0002672533-20200911-OSIRIS-OsirisSpectralFlat.fits'

hgar_r1000b = 'arc/0002672524-20200911-OSIRIS-OsirisCalibrationLamp.fits'
ne_r1000b_1 = 'arc/0002672525-20200911-OSIRIS-OsirisCalibrationLamp.fits'
ne_r1000b_2 = 'arc/0002672526-20200911-OSIRIS-OsirisCalibrationLamp.fits'

# Creates a master bias frame using median combine
bias_CCDData = []
bias_frames = [
    bias_1, bias_2, bias_3, bias_4, bias_5, bias_6, bias_7, bias_8, bias_9,
    bias_10, bias_11
]

for i in bias_frames:
    # Open all the bias frames
    bias = fits.open(os.path.join(base_folder, i))[2]
    bias_CCDData.append(CCDData(bias.data, unit=u.ct))

# Put data into a Combiner
bias_combiner = Combiner(bias_CCDData)

# Apply sigma clipping
bias_combiner.sigma_clipping(low_thresh=5.0, high_thresh=5.0, func=np.ma.mean)

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
    flat_r1000b_1, flat_r1000b_2, flat_r1000b_3, flat_r1000b_4
]

for i in flat_r1000b_frames:
    # Open all the flat frames
    flat_r1000b = fits.open(os.path.join(base_folder, i))[2]
    flat_r1000b_CCDData.append(CCDData(flat_r1000b.data, unit=u.ct))

# Put data into a Combiner
flat_r1000b_combiner = Combiner(flat_r1000b_CCDData)

# Apply sigma clipping
flat_r1000b_combiner.sigma_clipping(low_thresh=5.0,
                                    high_thresh=5.0,
                                    func=np.ma.median)

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
arc_r1000b_combiner.sigma_clipping(low_thresh=5.0,
                                   high_thresh=5.0,
                                   func=np.ma.median)

arc_r1000b_master = arc_r1000b_combiner.sum_combine()

# Free memory
arc_r1000b = None
arc_r1000b_CCDData = None
arc_r1000b_combiner = None

spatial_mask = np.arange(100, 1000)

light_fits_1 = fits.open(os.path.join(base_folder, light_r1000b_1))
light_fits_2 = fits.open(os.path.join(base_folder, light_r1000b_2))
standard_fits = fits.open(os.path.join(base_folder, standard_r1000b))

flat_master = flat_r1000b_master.data
flat_master -= bias_master.data
flat_master = signal.medfilt2d(flat_master)
flat_normalised = flat_master / np.nanmax(flat_master)

light_frame_1 = light_fits_1[2].data.astype('float')
light_frame_1 = detect_cosmics(light_frame_1 / 0.95,
                               gain=0.95,
                               readnoise=4.5,
                               fsmode='convolve',
                               psfmodel='gaussy',
                               psfsize=7)[1]
light_frame_1 -= bias_master.data
light_frame_1 /= flat_normalised

light_frame_2 = light_fits_2[2].data.astype('float')
light_frame_2 = detect_cosmics(light_frame_2 / 0.95,
                               gain=0.95,
                               readnoise=4.5,
                               fsmode='convolve',
                               psfmodel='gaussy',
                               psfsize=7)[1]
light_frame_2 -= bias_master.data
light_frame_2 /= flat_normalised

standard_frame = standard_fits[2].data.astype('float')
standard_frame = detect_cosmics(standard_frame / 0.95,
                                gain=0.95,
                                readnoise=4.5,
                                fsmode='convolve',
                                psfmodel='gaussy',
                                psfsize=7)[1]
standard_frame -= bias_master.data
standard_frame /= flat_normalised

science_1_twodspec = spectral_reduction.TwoDSpec(light_frame_1,
                                                 light_fits_1[0].header,
                                                 spatial_mask=spatial_mask,
                                                 saxis=0,
                                                 log_file_name=None,
                                                 log_level='INFO')

science_2_twodspec = spectral_reduction.TwoDSpec(light_frame_2,
                                                 light_fits_2[0].header,
                                                 spatial_mask=spatial_mask,
                                                 saxis=0,
                                                 log_file_name=None,
                                                 log_level='INFO')

standard_twodspec = spectral_reduction.TwoDSpec(standard_frame,
                                                standard_fits[0].header,
                                                spatial_mask=spatial_mask,
                                                saxis=0,
                                                log_file_name=None,
                                                log_level='INFO')

science_1_twodspec.ap_trace(display=True, fit_deg=3)
science_1_twodspec.ap_extract(display=True,
                              apwidth=9,
                              skywidth=9,
                              model='lowess')
science_1_twodspec.add_arc(arc_r1000b_master.data)
science_1_twodspec.apply_mask_to_arc()
science_1_twodspec.extract_arc_spec(display=False)

science_2_twodspec.ap_trace(display=True, fit_deg=3)
science_2_twodspec.ap_extract(display=True,
                              apwidth=9,
                              skywidth=9,
                              model='lowess')
science_2_twodspec.add_arc(arc_r1000b_master.data)
science_2_twodspec.apply_mask_to_arc()
science_2_twodspec.extract_arc_spec(display=False)

standard_twodspec.ap_trace(display=True, nspec=1, fit_deg=3)
standard_twodspec.ap_extract(display=True,
                             apwidth=9,
                             skywidth=9,
                             model='lowess')
standard_twodspec.add_arc(arc_r1000b_master.data)
standard_twodspec.apply_mask_to_arc()
standard_twodspec.extract_arc_spec(display=False)

# One dimensional spectral operation
science_1_onedspec = spectral_reduction.OneDSpec(log_file_name=None,
                                                 log_level='INFO')
science_1_onedspec.from_twodspec(science_1_twodspec, stype='science')
science_1_onedspec.from_twodspec(standard_twodspec, stype='standard')

science_2_onedspec = spectral_reduction.OneDSpec(log_file_name=None,
                                                 log_level='INFO')
science_2_onedspec.from_twodspec(science_2_twodspec, stype='science')
science_2_onedspec.from_twodspec(standard_twodspec, stype='standard')

science_1_onedspec.find_arc_lines(prominence=1,
                                  distance=3,
                                  refine_window_width=3,
                                  display=True)
science_2_onedspec.find_arc_lines(prominence=1,
                                  distance=3,
                                  refine_window_width=3,
                                  display=True)

atlas = [
    3650.153, 4046.563, 4077.831, 4358.328, 5460.735, 5769.598, 5790.663,
    6682.960, 6752.834, 6871.289, 6965.431, 7030.251, 7067.218, 7147.042,
    7272.936, 7383.981, 7503.869, 7514.652, 7635.106, 7723.98, 5852.488,
    5881.895, 5944.834, 5975.534, 6074.338, 6096.163, 6266.495, 6334.428,
    6402.248, 6506.528, 6532.822, 6598.953, 6678.276, 6717.043, 6929.467,
    7032.413, 7173.938, 7245.167, 7438.899, 7488.871, 7535.774
]
element = ['HgArNe'] * len(atlas)

# Configure the wavelength calibrator
science_1_onedspec.initialise_calibrator(stype='science+standard')

science_1_onedspec.set_hough_properties(num_slopes=2000,
                                        xbins=200,
                                        ybins=200,
                                        min_wavelength=3600,
                                        max_wavelength=7500,
                                        stype='science+standard')
science_1_onedspec.set_ransac_properties(filter_close=True,
                                         sample_size=6,
                                         stype='science+standard')

science_1_onedspec.add_user_atlas(elements=element,
                                  wavelengths=atlas,
                                  stype='science+standard')

science_1_onedspec.do_hough_transform()

# Solve for the pixel-to-wavelength solution
science_1_onedspec.fit(max_tries=2000, stype='science+standard', display=True)

# Apply the wavelength calibration and display it
science_1_onedspec.apply_wavelength_calibration(stype='science+standard')

# Get the standard from the library
science_1_onedspec.load_standard(target='feige110')

science_1_onedspec.compute_sensitivity(k=3,
                                       method='interpolate',
                                       mask_fit_size=5)
science_1_onedspec.inspect_sensitivity()

science_1_onedspec.apply_flux_calibration(stype='science+standard')

# Apply atmospheric extinction correction
science_1_onedspec.set_atmospheric_extinction(location='orm')
science_1_onedspec.apply_atmospheric_extinction_correction()

science_1_onedspec.inspect_reduced_spectrum()

# Configure the wavelength calibrator
science_2_onedspec.initialise_calibrator(stype='science+standard')

science_2_onedspec.set_hough_properties(num_slopes=2000,
                                        xbins=200,
                                        ybins=200,
                                        min_wavelength=3600,
                                        max_wavelength=7500,
                                        stype='science+standard')
science_2_onedspec.set_ransac_properties(filter_close=True,
                                         sample_size=6,
                                         stype='science+standard')

science_2_onedspec.add_user_atlas(elements=element,
                                  wavelengths=atlas,
                                  stype='science+standard')

science_2_onedspec.do_hough_transform()

# Solve for the pixel-to-wavelength solution
science_2_onedspec.fit(max_tries=2000, stype='science+standard', display=False)

# Apply the wavelength calibration and display it
science_2_onedspec.apply_wavelength_calibration(stype='science+standard')

# Get the standard from the library
science_2_onedspec.load_standard(target='feige110')

science_2_onedspec.compute_sensitivity(k=3,
                                       method='interpolate',
                                       mask_fit_size=5)
science_2_onedspec.inspect_sensitivity()

science_2_onedspec.apply_flux_calibration(stype='science+standard')

# Apply atmospheric extinction correction
science_2_onedspec.set_atmospheric_extinction(location='orm')
science_2_onedspec.apply_atmospheric_extinction_correction()

science_2_onedspec.inspect_reduced_spectrum()

wave_1 = science_1_onedspec.science_spectrum_list[0].wave_resampled
wave_1_bin2 = wave_1[::2]
wave_1_bin5 = wave_1[::5]
wave_1_bin10 = wave_1[::10]

flux_1 = science_1_onedspec.science_spectrum_list[0].flux_resampled
flux_1_bin2 = spectres(wave_1_bin2, wave_1, flux_1)
flux_1_bin5 = spectres(wave_1_bin5, wave_1, flux_1)
flux_1_bin10 = spectres(wave_1_bin10, wave_1, flux_1)

wave_2 = science_2_onedspec.science_spectrum_list[0].wave_resampled

flux_2 = science_2_onedspec.science_spectrum_list[0].flux_resampled
# note this is resampled to match wave_1s
flux_2_bin2 = spectres(wave_1_bin2, wave_2, flux_2)
flux_2_bin5 = spectres(wave_1_bin5, wave_2, flux_2)
flux_2_bin10 = spectres(wave_1_bin10, wave_2, flux_2)

# Get the interstellar dust extinction correction
ext_bin2 = extinction.fm07(wave_1_bin2, 0.8 * 3.1)
ext_bin5 = extinction.fm07(wave_1_bin5, 0.8 * 3.1)

#
#
# Process the R2500U here
#
#

light_r2500u_1 =\
    'object/0002673193-20200913-OSIRIS-OsirisLongSlitSpectroscopy.fits'
light_r2500u_2 =\
    'object/0002673194-20200913-OSIRIS-OsirisLongSlitSpectroscopy.fits'

standard_r2500u =\
    'stds/0002673245-20200913-OSIRIS-OsirisLongSlitSpectroscopy.fits'

flat_r2500u_1 = 'flat/0002672538-20200911-OSIRIS-OsirisSpectralFlat.fits'
flat_r2500u_2 = 'flat/0002672539-20200911-OSIRIS-OsirisSpectralFlat.fits'
flat_r2500u_3 = 'flat/0002672540-20200911-OSIRIS-OsirisSpectralFlat.fits'
flat_r2500u_4 = 'flat/0002672541-20200911-OSIRIS-OsirisSpectralFlat.fits'

hgar_r2500u = 'arc/0002672534-20200911-OSIRIS-OsirisCalibrationLamp.fits'
xe_r2500u = 'arc/0002672535-20200911-OSIRIS-OsirisCalibrationLamp.fits'
xe_r2500u = 'arc/0002672536-20200911-OSIRIS-OsirisCalibrationLamp.fits'

# Creates a master flat frame using median combine
flat_r2500u_CCDData = []
flat_r2500u_frames = [
    flat_r2500u_1, flat_r2500u_2, flat_r2500u_3, flat_r2500u_4
]

for i in flat_r2500u_frames:
    # Open all the flat frames
    flat_r2500u = fits.open(os.path.join(base_folder, i))[2]
    flat_r2500u_CCDData.append(CCDData(flat_r2500u.data, unit=u.ct))

# Put data into a Combiner
flat_r2500u_combiner = Combiner(flat_r2500u_CCDData)

# Apply sigma clipping
flat_r2500u_combiner.sigma_clipping(low_thresh=5.0,
                                    high_thresh=5.0,
                                    func=np.ma.median)

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
arc_r2500u_combiner.sigma_clipping(low_thresh=5.0,
                                   high_thresh=5.0,
                                   func=np.ma.median)

arc_r2500u_master = arc_r2500u_combiner.sum_combine()

# Free memory
arc_r2500u = None
arc_r2500u_CCDData = None
arc_r2500u_combiner = None

spatial_mask_r2500u = np.arange(50, 1000)

light_fits_r2500u_1 = fits.open(os.path.join(base_folder, light_r2500u_1))
light_fits_r2500u_2 = fits.open(os.path.join(base_folder, light_r2500u_2))
standard_fits_r2500u = fits.open(os.path.join(base_folder, standard_r2500u))

flat_r2500u_master = flat_r2500u_master.data
flat_r2500u_master -= bias_master.data
flat_r2500u_master = signal.medfilt2d(flat_r2500u_master)
flat_r2500u_normalised = flat_r2500u_master / np.nanmax(flat_r2500u_master)

light_frame_r2500u_1 = light_fits_r2500u_1[2].data.astype('float')
light_frame_r2500u_1 = detect_cosmics(light_frame_r2500u_1 / 0.95,
                                      gain=0.95,
                                      readnoise=4.5,
                                      fsmode='convolve',
                                      psfmodel='gaussy',
                                      psfsize=7)[1]
light_frame_r2500u_1 -= bias_master.data
light_frame_r2500u_1 /= flat_r2500u_normalised

light_frame_r2500u_2 = light_fits_r2500u_2[2].data.astype('float')
light_frame_r2500u_2 = detect_cosmics(light_frame_r2500u_2 / 0.95,
                                      gain=0.95,
                                      readnoise=4.5,
                                      fsmode='convolve',
                                      psfmodel='gaussy',
                                      psfsize=7)[1]
light_frame_r2500u_2 -= bias_master.data
light_frame_r2500u_2 /= flat_r2500u_normalised

standard_frame_r2500u = standard_fits_r2500u[2].data.astype('float')
standard_frame_r2500u = detect_cosmics(standard_frame_r2500u / 0.95,
                                       gain=0.95,
                                       readnoise=4.5,
                                       fsmode='convolve',
                                       psfmodel='gaussy',
                                       psfsize=7)[1]
standard_frame_r2500u -= bias_master.data
standard_frame_r2500u /= flat_r2500u_normalised

science_r2500u_1_twodspec = spectral_reduction.TwoDSpec(
    light_frame_r2500u_1,
    light_fits_r2500u_1[0].header,
    spatial_mask=spatial_mask_r2500u,
    saxis=0,
    log_file_name=None,
    log_level='INFO')

science_r2500u_2_twodspec = spectral_reduction.TwoDSpec(
    light_frame_r2500u_2,
    light_fits_r2500u_2[0].header,
    spatial_mask=spatial_mask_r2500u,
    saxis=0,
    log_file_name=None,
    log_level='INFO')

standard_r2500u_twodspec = spectral_reduction.TwoDSpec(
    standard_frame_r2500u,
    standard_fits_r2500u[0].header,
    spatial_mask=spatial_mask_r2500u,
    saxis=0,
    log_file_name=None,
    log_level='INFO')

science_r2500u_1_twodspec.ap_trace(display=True, fit_deg=3)
science_r2500u_1_twodspec.ap_extract(display=True,
                                     apwidth=9,
                                     skywidth=9,
                                     model='lowess')
science_r2500u_1_twodspec.add_arc(arc_r2500u_master.data)
science_r2500u_1_twodspec.apply_mask_to_arc()
science_r2500u_1_twodspec.extract_arc_spec(display=False)

science_r2500u_2_twodspec.ap_trace(display=True, fit_deg=3)
science_r2500u_2_twodspec.ap_extract(display=True,
                                     apwidth=9,
                                     skywidth=9,
                                     model='lowess')
science_r2500u_2_twodspec.add_arc(arc_r2500u_master.data)
science_r2500u_2_twodspec.apply_mask_to_arc()
science_r2500u_2_twodspec.extract_arc_spec(display=False)

# saturated
standard_r2500u_twodspec.ap_trace(display=True, nspec=1, fit_deg=3)
standard_r2500u_twodspec.spectrum_list[
    0].trace_sigma = science_r2500u_2_twodspec.spectrum_list[0].trace_sigma
standard_r2500u_twodspec.ap_extract(display=True,
                                    apwidth=9,
                                    skywidth=9,
                                    model='lowess')
standard_r2500u_twodspec.add_arc(arc_r2500u_master.data)
standard_r2500u_twodspec.apply_mask_to_arc()
standard_r2500u_twodspec.extract_arc_spec(display=False)

# One dimensional spectral operation
science_r2500u_1_onedspec = spectral_reduction.OneDSpec(log_file_name=None,
                                                        log_level='INFO')
science_r2500u_1_onedspec.from_twodspec(science_r2500u_1_twodspec,
                                        stype='science')
science_r2500u_1_onedspec.from_twodspec(standard_r2500u_twodspec,
                                        stype='standard')

science_r2500u_2_onedspec = spectral_reduction.OneDSpec(log_file_name=None,
                                                        log_level='INFO')
science_r2500u_2_onedspec.from_twodspec(science_r2500u_2_twodspec,
                                        stype='science')
science_r2500u_2_onedspec.from_twodspec(standard_r2500u_twodspec,
                                        stype='standard')

science_r2500u_1_onedspec.find_arc_lines(prominence=1.25,
                                         distance=3,
                                         refine_window_width=3,
                                         display=True)
science_r2500u_2_onedspec.find_arc_lines(prominence=1.25,
                                         distance=3,
                                         refine_window_width=3,
                                         display=True)

atlas_r2500u = [
    3650.153, 4046.563, 4077.831, 4358.328, 4500.977, 4524.680, 4582.747
]

# 4624.276, 4671.226, 4697.02, 4734.152, 4807.00687, 4829.69549, 4843.33135,
# 4916.52629, 4923.13617, 5460.735
element_r2500u = ['HgArXe'] * len(atlas_r2500u)

# Configure the wavelength calibrator
science_r2500u_1_onedspec.initialise_calibrator(stype='science+standard')

science_r2500u_1_onedspec.set_hough_properties(num_slopes=1000,
                                               xbins=100,
                                               ybins=100,
                                               min_wavelength=3200,
                                               max_wavelength=4800,
                                               stype='science+standard')
science_r2500u_1_onedspec.set_ransac_properties(filter_close=True,
                                                sample_size=3,
                                                stype='science+standard')

science_r2500u_1_onedspec.add_user_atlas(elements=element_r2500u,
                                         wavelengths=atlas_r2500u,
                                         stype='science+standard')

science_r2500u_1_onedspec.do_hough_transform()

# Solve for the pixel-to-wavelength solution
science_r2500u_1_onedspec.fit(max_tries=2000,
                              stype='science+standard',
                              display=True)

# Apply the wavelength calibration and display it
science_r2500u_1_onedspec.apply_wavelength_calibration(
    stype='science+standard')

# Get the standard from the library
science_r2500u_1_onedspec.load_standard(target='feige110')

science_r2500u_1_onedspec.compute_sensitivity(k=3,
                                              method='interpolate',
                                              mask_fit_size=5)
science_r2500u_1_onedspec.inspect_sensitivity()

science_r2500u_1_onedspec.apply_flux_calibration(stype='science+standard')

# Apply atmospheric extinction correction
science_r2500u_1_onedspec.set_atmospheric_extinction(location='orm')
science_r2500u_1_onedspec.apply_atmospheric_extinction_correction()

science_r2500u_1_onedspec.inspect_reduced_spectrum(wave_min=3200.,
                                                   wave_max=4800.)

# Configure the wavelength calibrator
science_r2500u_2_onedspec.initialise_calibrator(stype='science+standard')

science_r2500u_2_onedspec.set_hough_properties(num_slopes=1000,
                                               xbins=100,
                                               ybins=100,
                                               min_wavelength=3000,
                                               max_wavelength=4800,
                                               stype='science+standard')
science_r2500u_2_onedspec.set_ransac_properties(filter_close=True,
                                                sample_size=3,
                                                stype='science+standard')

science_r2500u_2_onedspec.add_user_atlas(elements=element_r2500u,
                                         wavelengths=atlas_r2500u,
                                         stype='science+standard')

science_r2500u_2_onedspec.do_hough_transform()

# Solve for the pixel-to-wavelength solution
science_r2500u_2_onedspec.fit(max_tries=2000,
                              stype='science+standard',
                              display=False)

# Apply the wavelength calibration and display it
science_r2500u_2_onedspec.apply_wavelength_calibration(
    stype='science+standard')

# Get the standard from the library
science_r2500u_2_onedspec.load_standard(target='feige110')

science_r2500u_2_onedspec.compute_sensitivity(k=3,
                                              method='interpolate',
                                              mask_fit_size=5)
science_r2500u_2_onedspec.inspect_sensitivity()

science_r2500u_2_onedspec.apply_flux_calibration(stype='science+standard')

# Apply atmospheric extinction correction
science_r2500u_2_onedspec.set_atmospheric_extinction(location='orm')
science_r2500u_2_onedspec.apply_atmospheric_extinction_correction()

science_r2500u_2_onedspec.inspect_reduced_spectrum(wave_min=3000.,
                                                   wave_max=4800.)

wave_r2500u_1 = science_r2500u_1_onedspec.science_spectrum_list[
    0].wave_resampled
wave_r2500u_1_bin2 = wave_r2500u_1[::2]
wave_r2500u_1_bin5 = wave_r2500u_1[::5]
wave_r2500u_1_bin10 = wave_r2500u_1[::10]

flux_r2500u_1 = science_r2500u_1_onedspec.science_spectrum_list[
    0].flux_resampled
flux_r2500u_1_bin2 = spectres(wave_r2500u_1_bin2, wave_r2500u_1, flux_r2500u_1)
flux_r2500u_1_bin5 = spectres(wave_r2500u_1_bin5, wave_r2500u_1, flux_r2500u_1)
flux_r2500u_1_bin10 = spectres(wave_r2500u_1_bin10, wave_r2500u_1,
                               flux_r2500u_1)

wave_r2500u_2 = science_r2500u_2_onedspec.science_spectrum_list[
    0].wave_resampled

flux_r2500u_2 = science_r2500u_2_onedspec.science_spectrum_list[
    0].flux_resampled
# note this is resampled to match wave_1s
flux_r2500u_2_bin2 = spectres(wave_r2500u_1_bin2, wave_r2500u_2, flux_r2500u_2)
flux_r2500u_2_bin5 = spectres(wave_r2500u_1_bin5, wave_r2500u_2, flux_r2500u_2)
flux_r2500u_2_bin10 = spectres(wave_r2500u_1_bin10, wave_r2500u_2,
                               flux_r2500u_2)

# Get the interstellar dust extinction correction
ext_r2500u_bin2 = extinction.fm07(wave_r2500u_1_bin2, 0.8 * 3.1)
ext_r2500u_bin5 = extinction.fm07(wave_r2500u_1_bin5, 0.8 * 3.1)

ymax = np.nanmax(extinction.remove(ext_r2500u_bin2, flux_r2500u_1_bin2))

plt.figure(1, figsize=(12, 5))
plt.clf()
plt.plot(wave_1_bin2,
         extinction.remove(ext_bin2, flux_1_bin2),
         lw=0.5,
         label='R1000B epoch 1')
plt.plot(wave_1_bin2,
         extinction.remove(ext_bin2, flux_2_bin2),
         lw=0.5,
         label='R1000B epoch 2')
plt.plot(wave_1_bin2,
         extinction.remove(ext_bin2,
                           np.nanmean((flux_1_bin2, flux_2_bin2), axis=0)),
         color='black',
         label='R1000B average')
plt.plot(wave_r2500u_1_bin2,
         extinction.remove(ext_r2500u_bin2, flux_r2500u_1_bin2),
         lw=0.5,
         label='R2500U epoch 1')
plt.plot(wave_r2500u_1_bin2,
         extinction.remove(ext_r2500u_bin2, flux_r2500u_2_bin2),
         lw=0.5,
         label='R2500U epoch 2')
plt.plot(wave_r2500u_1_bin2,
         extinction.remove(
             ext_r2500u_bin2,
             np.nanmean((flux_r2500u_1_bin2, flux_r2500u_2_bin2), axis=0)),
         color='purple',
         label='R2500U average')
plt.vlines(lines_H, 0, ymax, color='blue', label='H')
plt.vlines(lines_HeI, 0, ymax, color='red', label='He I')
plt.vlines(lines_HeII, 0, ymax, color='green', label='He II')
plt.xlim(3400, 8000)
plt.ylim(0, 8e-14)
plt.xlabel('Wavelength / A')
plt.ylabel('Flux / ( erg / cm / cm / s / A)')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('ZTF_BLAP_01_GTC_bin2.png')

plt.figure(2, figsize=(12, 5))
plt.clf()
plt.plot(wave_1_bin5,
         extinction.remove(ext_bin5, flux_1_bin5),
         lw=0.5,
         label='R1000B epoch 1')
plt.plot(wave_1_bin5,
         extinction.remove(ext_bin5, flux_2_bin5),
         lw=0.5,
         label='R1000B epoch 2')
plt.plot(wave_1_bin5,
         extinction.remove(ext_bin5,
                           np.nanmean((flux_1_bin5, flux_2_bin5), axis=0)),
         color='black',
         label='R1000B average')
plt.plot(wave_r2500u_1_bin5,
         extinction.remove(ext_r2500u_bin5, flux_r2500u_1_bin5),
         lw=0.5,
         label='R2500U epoch 1')
plt.plot(wave_r2500u_1_bin5,
         extinction.remove(ext_r2500u_bin5, flux_r2500u_2_bin5),
         lw=0.5,
         label='R2500U epoch 2')
plt.plot(wave_r2500u_1_bin5,
         extinction.remove(
             ext_r2500u_bin5,
             np.nanmean((flux_r2500u_1_bin5, flux_r2500u_2_bin5), axis=0)),
         color='purple',
         label='R2500U average')
plt.vlines(lines_H, 0, ymax, color='blue', label='H')
plt.vlines(lines_HeI, 0, ymax, color='red', label='He I')
plt.vlines(lines_HeII, 0, ymax, color='green', label='He II')
plt.xlim(3400, 8000)
plt.ylim(0, 8e-14)
plt.xlabel('Wavelength / A')
plt.ylabel('Flux / ( erg / cm / cm / s / A)')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('ZTF_BLAP_01_GTC_bin5.png')

np.save('wavelength_R1000B', wave_1_bin2)
np.save(
    'flux_R1000U',
    extinction.remove(ext_bin2, np.nanmean((flux_1_bin2, flux_2_bin2),
                                           axis=0)))

np.save('wavelength_R2500U', wave_r2500u_1_bin2)
np.save(
    'flux_R2500U',
    extinction.remove(
        ext_r2500u_bin2,
        np.nanmean((flux_r2500u_1_bin2, flux_r2500u_2_bin2), axis=0)))
