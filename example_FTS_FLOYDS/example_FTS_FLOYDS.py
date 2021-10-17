import copy

import numpy as np
from astropy.io import fits
from aspired import spectral_reduction
from matplotlib import pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

# Line list
atlas_Hg_red = [5460.7348, 5769.5982, 5790.6630]
atlas_Ar_red = [
    6965.4307, 7067.2175, 7147.0416, 7272.9359, 7383.9805, 7503.8691,
    7635.1056, 7723.7599, 7948.1764, 8014.7857, 8115.3108, 8264.5225,
    8424.6475, 8521.4422, 8667.944, 9122.9674, 9224.4992, 9354.220, 9657.7863,
    9784.503
]
element_Hg_red = ['Hg'] * len(atlas_Hg_red)
element_Ar_red = ['Ar'] * len(atlas_Ar_red)

atlas_Ar_blue = [4158.590, 4200.674, 4272.169, 4300.101, 4510.733]
atlas_Hg_blue = [
    3650.153, 4046.563, 4077.8314, 4358.328, 5460.7348, 5769.598, 5790.663
]
atlas_Zn_blue = [4722.1569]
element_Hg_blue = ['Hg'] * len(atlas_Hg_blue)
element_Ar_blue = ['Ar'] * len(atlas_Ar_blue)
element_Zn_blue = ['Zn'] * len(atlas_Zn_blue)

# Set the frame
red_spatial_mask = np.arange(0, 330)
blue_spatial_mask = np.arange(335, 512)
red_spec_mask = np.arange(0, 1800)
blue_spec_mask = np.arange(500, 2060)


def extract_floyds(light_fits,
                   flat_fits,
                   arc_fits,
                   coeff_red=None,
                   coeff_blue=None):
    light_data = light_fits.data
    light_header = fits.Header(light_fits.header)

    red = spectral_reduction.TwoDSpec(light_data,
                                      header=light_header,
                                      spatial_mask=red_spatial_mask,
                                      spec_mask=red_spec_mask,
                                      cosmicray=True,
                                      sigclip=2.,
                                      readnoise=3.5,
                                      gain=2.3,
                                      log_level='INFO',
                                      log_file_name=None)

    blue = spectral_reduction.TwoDSpec(light_data,
                                       header=light_header,
                                       spatial_mask=blue_spatial_mask,
                                       spec_mask=blue_spec_mask,
                                       cosmicray=True,
                                       sigclip=2.,
                                       readnoise=3.5,
                                       gain=2.3,
                                       log_level='INFO',
                                       log_file_name=None)

    # Add the arcs before rectifying the image, which will apply the
    # rectification to the arc frames too
    blue.add_arc(arc_fits.data, fits.Header(arc_fits.header))
    blue.apply_mask_to_arc()
    red.add_arc(arc_fits.data, fits.Header(arc_fits.header))
    red.apply_mask_to_arc()

    # Get the trace to rectify the image
    red.ap_trace(nspec=1,
                 ap_faint=20,
                 percentile=10,
                 trace_width=50,
                 shift_tol=35,
                 fit_deg=5,
                 display=False)
    red.get_rectification(upsample_factor=5, coeff=coeff_red, display=True)
    red.apply_rectification()
    # Need to store the traces for fringe correction before overwriting them
    # with the new traces
    trace_red = copy.deepcopy(red.spectrum_list[0].trace)
    trace_sigma_red = copy.deepcopy(red.spectrum_list[0].trace_sigma)

    # Get the trace again for the rectified image and then extract
    red.ap_trace(nspec=1,
                 ap_faint=20,
                 percentile=10,
                 trace_width=20,
                 fit_deg=3,
                 display=False)
    red.ap_extract(apwidth=10, spec_id=0, display=False)

    # Do the same with the blue
    blue.ap_trace(nspec=1,
                  ap_faint=10,
                  percentile=10,
                  trace_width=20,
                  shift_tol=50,
                  fit_deg=5,
                  display=False)
    blue.get_rectification(upsample_factor=5, coeff=coeff_blue, display=True)
    blue.apply_rectification()

    blue.ap_trace(nspec=1,
                  percentile=10,
                  trace_width=20,
                  fit_deg=3,
                  display=False)
    blue.ap_extract(apwidth=10, display=False)

    red.extract_arc_spec(spec_width=15, display=False)
    blue.extract_arc_spec(spec_width=15, display=False)

    # Get the blue and red traces for fringe removal
    trace_red_rectified = red.spectrum_list[0].trace
    trace_sigma_red_rectified = red.spectrum_list[0].trace_sigma

    # Extract the red flat
    flat_data = flat_fits.data
    flat_header = fits.Header(flat_fits.header)

    flat = spectral_reduction.TwoDSpec(flat_data,
                                       header=flat_header,
                                       spatial_mask=red_spatial_mask,
                                       spec_mask=red_spec_mask,
                                       cosmicray=True,
                                       readnoise=3.5,
                                       gain=2.3,
                                       log_level='INFO',
                                       log_file_name=None)

    # Force extraction from the flat for fringe correction
    flat.add_trace(trace_red, trace_sigma_red)
    flat.get_rectification(coeff=red.rec_coeff)
    flat.apply_rectification()
    flat.add_trace(trace_red_rectified, trace_sigma_red_rectified)
    flat.ap_extract(apwidth=10, skywidth=0, display=False)

    return red, blue, flat


# only red specs are used
def fringe_correction(target, flat):
    flat_count = flat.spectrum_list[0].count
    flat_continuum = lowess(flat_count,
                            np.arange(len(flat_count)),
                            frac=0.04,
                            return_sorted=False)
    flat_continuum_divided = flat_count / flat_continuum
    flat_continuum_divided[:500] = 1.0
    # Apply the flat correction
    target.spectrum_list[0].count /= flat_continuum_divided


def calibrate_red(science, standard, standard_name, standard_library=None):
    #
    # Start handling 1D spectra here
    #
    # Need to add fringe subtraction here
    red_onedspec = spectral_reduction.OneDSpec(log_level='INFO',
                                               log_file_name=None)

    # Red spectrum first
    red_onedspec.from_twodspec(standard, stype='standard')
    red_onedspec.from_twodspec(science, stype='science')

    # Find the peaks of the arc
    red_onedspec.find_arc_lines(display=True,
                                prominence=0.25,
                                top_n_peaks=25,
                                stype='science+standard')

    # Configure the wavelength calibrator
    red_onedspec.initialise_calibrator(stype='science+standard')

    red_onedspec.add_user_atlas(elements=element_Hg_red,
                                wavelengths=atlas_Hg_red,
                                stype='science+standard')
    red_onedspec.add_user_atlas(elements=element_Ar_red,
                                wavelengths=atlas_Ar_red,
                                stype='science+standard')

    red_onedspec.set_hough_properties(num_slopes=2000,
                                      xbins=200,
                                      ybins=200,
                                      min_wavelength=4750,
                                      max_wavelength=11750,
                                      stype='science+standard')
    red_onedspec.set_ransac_properties(sample_size=8,
                                       minimum_matches=15,
                                       stype='science+standard')
    red_onedspec.do_hough_transform(stype='science+standard')

    # Solve for the pixel-to-wavelength solution
    red_onedspec.fit(max_tries=5000, stype='science+standard', display=True)

    # Apply the wavelength calibration and display it
    red_onedspec.apply_wavelength_calibration(wave_start=5000,
                                              wave_end=11000,
                                              wave_bin=1,
                                              stype='science+standard')

    red_onedspec.load_standard(standard_name, standard_library)
    red_onedspec.get_sensitivity(mask_range=[[6850, 6960], [7576, 7680]])
    red_onedspec.inspect_sensitivity()
    red_onedspec.apply_flux_calibration()
    red_onedspec.get_telluric_profile()
    red_onedspec.inspect_telluric_profile()
    red_onedspec.apply_telluric_correction()
    red_onedspec.apply_atmospheric_extinction_correction()

    return red_onedspec


def calibrate_blue(science, standard, standard_name, standard_library=None):
    # Blue spectrum here
    blue = spectral_reduction.OneDSpec(log_level='INFO', log_file_name=None)

    blue.from_twodspec(standard, stype='standard')
    blue.from_twodspec(science, stype='science')

    blue.find_arc_lines(prominence=0.25,
                        top_n_peaks=10,
                        display=True,
                        stype='science+standard')

    blue.initialise_calibrator(stype='science+standard')

    blue.add_user_atlas(elements=element_Hg_blue,
                        wavelengths=atlas_Hg_blue,
                        stype='science+standard')
    blue.add_user_atlas(elements=element_Ar_blue,
                        wavelengths=atlas_Ar_blue,
                        stype='science+standard')
    blue.add_user_atlas(elements=element_Zn_blue,
                        wavelengths=atlas_Zn_blue,
                        stype='science+standard')

    blue.set_hough_properties(num_slopes=2000,
                              xbins=200,
                              ybins=200,
                              min_wavelength=3000,
                              max_wavelength=6000,
                              stype='science+standard')
    blue.set_ransac_properties(sample_size=8,
                               filter_close=True,
                               stype='science+standard')
    blue.do_hough_transform(stype='science+standard')

    # Solve for the pixel-to-wavelength solution
    blue.fit(max_tries=2000, stype='science+standard', display=True)

    # Apply the wavelength calibration and display it
    blue.apply_wavelength_calibration(wave_start=3200,
                                      wave_end=5600,
                                      wave_bin=1,
                                      stype='science+standard')

    blue.load_standard(standard_name, standard_library)
    blue.get_sensitivity()
    blue.inspect_sensitivity()
    blue.apply_flux_calibration()

    return blue


#
# Standard frame here
#
standard_light_fits = fits.open("FLOYDS_AT2019mtw_raw_data/standard/"
                                "coj2m002-en12-20210126-0003-e00.fits.fz")[1]

standard_flat_fits = fits.open("FLOYDS_AT2019mtw_raw_data/standard/"
                               "coj2m002-en12-20210126-0004-w00.fits.fz")[1]

standard_arc_fits = fits.open("FLOYDS_AT2019mtw_raw_data/standard/"
                              "coj2m002-en12-20210126-0005-a00.fits.fz")[1]

standard_twodspec_red, standard_twodspec_blue, standard_twodspec_flat =\
    extract_floyds(standard_light_fits, standard_flat_fits, standard_arc_fits)

#
# Science frame here
#
science_light_fits = fits.open("FLOYDS_AT2019mtw_raw_data/science/"
                               "coj2m002-en12-20200321-0021-e00.fits.fz")[1]

science_flat_fits = fits.open("FLOYDS_AT2019mtw_raw_data/science/"
                              "coj2m002-en12-20200321-0023-w00.fits.fz")[1]

science_arc_fits = fits.open("FLOYDS_AT2019mtw_raw_data/science/"
                             "coj2m002-en12-20200321-0022-a00.fits.fz")[1]

AT2019mtw_twodspec_red, AT2019mtw_twodspec_blue, AT2019mtw_twodspec_flat =\
    extract_floyds(
        science_light_fits,
        science_flat_fits,
        science_arc_fits)

standard_name = 'l74546a'
standard_library = 'irafredcal'

# Fringe Correction
fringe_correction(AT2019mtw_twodspec_red, AT2019mtw_twodspec_flat)
fringe_correction(standard_twodspec_red, standard_twodspec_flat)

# Start working in 1D
onedspec_blue = calibrate_blue(AT2019mtw_twodspec_blue, standard_twodspec_blue,
                               standard_name)
onedspec_red = calibrate_red(AT2019mtw_twodspec_red, standard_twodspec_red,
                             standard_name, standard_library)

# Inspect
onedspec_blue.inspect_reduced_spectrum(wave_min=3000,
                                       wave_max=6000,
                                       stype='science+standard')

onedspec_red.inspect_reduced_spectrum(wave_min=5000,
                                      wave_max=11000,
                                      stype='science+standard')

official_fits = fits.open(
    'FLOYDS_AT2019mtw_raw_data/science/'
    'TAU2020A-010_2092923_fts_20200321_58930/'
    'nttAT2019mtw_fts_20200321_merge_2.0_58930_1_2df_ex.fits')[0]
wave_start = float(official_fits.header['CRVAL1'])
wave_bin = float(official_fits.header['CD1_1'])
spec_length = len(official_fits.data[0][0])
wave = np.linspace(wave_start, wave_start + wave_bin * spec_length,
                   spec_length)

LCO_unit = 1e-20

plt.figure(100, figsize=(16, 8))
plt.clf()
plt.plot(onedspec_blue.science_spectrum_list[0].wave_resampled,
         onedspec_blue.science_spectrum_list[0].flux_resampled,
         label='Blue arm (ASPIRED)')
plt.plot(onedspec_red.science_spectrum_list[0].wave_resampled,
         onedspec_red.science_spectrum_list[0].flux_resampled,
         label='Red arm (ASPIRED)')
plt.plot(wave,
         official_fits.data[0][0] * LCO_unit,
         color='black',
         label='Official LCO Pipeline')
plt.xlim(3000., 10000.)
plt.ylim(0, max(official_fits.data[0][0]) * LCO_unit * 1.05)
plt.xlabel('Wavelength / A')
plt.ylabel('Flux / (erg / s / cm / cm / A)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.title('AT 2019 MTW')
plt.savefig('AT2019mtw.png')

redshift = 0.06707
wave_rest = wave / (1 + redshift)
wave_red_rest = onedspec_red.science_spectrum_list[0].wave_resampled / (
    1 + redshift)
wave_blue_rest = onedspec_blue.science_spectrum_list[0].wave_resampled / (
    1 + redshift)

plt.figure(200, figsize=(16, 8))
plt.clf()
plt.plot(wave_blue_rest,
         onedspec_blue.science_spectrum_list[0].flux_resampled,
         label='Blue arm (ASPIRED)')
plt.plot(wave_red_rest,
         onedspec_red.science_spectrum_list[0].flux_resampled,
         label='Red arm (ASPIRED)')
plt.plot(wave_rest,
         official_fits.data[0][0] * LCO_unit,
         color='black',
         label='Official LCO Pipeline')

snex_fits = fits.open(
    '/Users/cylam/Downloads/AT2019mtw_20200321_redblu_181426.411.fits')[0]
wave_bin2 = float(snex_fits.header['CD1_1'])
wave_start2 = float(
    snex_fits.header['CRVAL1']) - float(snex_fits.header['CRPIX1']) * wave_bin2
spec_length2 = len(snex_fits.data[0][0])
wave2 = np.linspace(wave_start2, wave_start2 + wave_bin2 * spec_length2,
                    spec_length2) / (1 + redshift)
plt.plot(wave2, snex_fits.data[0][0] * 4, color='red', label='SNEx')

plt.xlim(3000., 10000.)
plt.ylim(
    0,
    max(onedspec_red.science_spectrum_list[0].flux_resampled[
        (wave_red_rest > 3500.) & (wave_red_rest < 10000.)]) * 1.05)
plt.xlabel('Rest Wavelength / A')
plt.ylabel('Flux / (erg / s / cm / cm / A)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.title('AT 2019 MTW (Rest Wavelength)')
plt.show()
plt.savefig('AT2019mtw_rest_wavelength.png')
