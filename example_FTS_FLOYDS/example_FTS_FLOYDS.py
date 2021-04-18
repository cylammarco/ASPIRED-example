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
    8424.6475, 8521.4422, 9122.9674, 9224.4992, 9657.7863
]
element_Hg_red = ['Hg'] * len(atlas_Hg_red)
element_Ar_red = ['Ar'] * len(atlas_Ar_red)

atlas_Hg_blue = [
    3650.153, 4046.563, 4077.8314, 4358.328, 4916.068, 5460.7348, 5769.5982
]
atlas_Zn_blue = [4078.14, 4298.3249, 4722.15, 4810.53, 5181.9819]
element_Hg_blue = ['Hg'] * len(atlas_Hg_blue)
element_Zn_blue = ['Zn'] * len(atlas_Zn_blue)

# Set the frame
red_spatial_mask = np.arange(0, 330)
blue_spatial_mask = np.arange(335, 512)
red_spec_mask = np.arange(0, 1900)
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
                                      log_file_name='None')

    blue = spectral_reduction.TwoDSpec(light_data,
                                       header=light_header,
                                       spatial_mask=blue_spatial_mask,
                                       spec_mask=blue_spec_mask,
                                       cosmicray=True,
                                       sigclip=2.,
                                       readnoise=3.5,
                                       gain=2.3,
                                       log_level='INFO',
                                       log_file_name='None')

    # Add the arcs before rectifying the image, which will apply the
    # rectification to the arc frames too
    blue.add_arc(arc_fits.data, fits.Header(arc_fits.header))
    blue.apply_twodspec_mask_to_arc()
    red.add_arc(arc_fits.data, fits.Header(arc_fits.header))
    red.apply_twodspec_mask_to_arc()

    # Get the trace to rectify the image
    red.ap_trace(nspec=1,
                 ap_faint=20,
                 trace_width=20,
                 shift_tol=50,
                 fit_deg=5,
                 display=True)
    red.compute_rectification(upsample_factor=10, coeff=coeff_red)
    red.apply_rectification()
    # Need to store the traces for fringe correction before overwriting them
    # with the new traces
    trace_red = red.spectrum_list[0].trace
    trace_sigma_red = red.spectrum_list[0].trace_sigma

    # Get the trace again for the rectified image and then extract
    red.ap_trace(nspec=1, trace_width=20, fit_deg=3, display=False)
    red.ap_extract(apwidth=10, spec_id=0, display=True)

    # Do the same with the blue
    blue.ap_trace(nspec=1,
                  ap_faint=20,
                  trace_width=20,
                  shift_tol=50,
                  fit_deg=5,
                  display=True)
    blue.compute_rectification(upsample_factor=10,
                               bin_size=7,
                               n_bin=[3, 1],
                               coeff=coeff_blue)
    blue.apply_rectification()

    blue.ap_trace(nspec=1,
                  percentile=30,
                  trace_width=20,
                  fit_deg=3,
                  display=True)
    blue.ap_extract(apwidth=10, display=True)

    red.extract_arc_spec(spec_width=15, display=True)
    blue.extract_arc_spec(spec_width=15, display=True)

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
                                       log_file_name='None')

    # Force extraction from the flat for fringe correction
    flat.add_trace(trace_red, trace_sigma_red)
    flat.compute_rectification(coeff=coeff_red)
    flat.apply_rectification()
    flat.add_trace(trace_red_rectified, trace_sigma_red_rectified)
    flat.ap_extract(apwidth=10, skywidth=0, display=True)

    return red, blue, flat


def calibrate_red(science, standard, standard_name):
    #
    # Start handling 1D spectra here
    #
    # Need to add fringe subtraction here
    red = spectral_reduction.OneDSpec(log_level='INFO', log_file_name=None)

    # Red spectrum first
    red.from_twodspec(standard, stype='standard')
    red.from_twodspec(science, stype='science')

    # Find the peaks of the arc
    red.find_arc_lines(display=True, prominence=100, stype='science')
    red.find_arc_lines(display=True, prominence=20, stype='standard')

    # Configure the wavelength calibrator
    red.initialise_calibrator(stype='science+standard')

    red.add_user_atlas(elements=element_Hg_red,
                       wavelengths=atlas_Hg_red,
                       stype='science+standard')
    red.add_user_atlas(elements=element_Ar_red,
                       wavelengths=atlas_Ar_red,
                       stype='science+standard')

    red.set_hough_properties(num_slopes=2000,
                             xbins=100,
                             ybins=100,
                             min_wavelength=4500,
                             max_wavelength=10500,
                             stype='science+standard')
    red.set_ransac_properties(stype='science+standard')
    red.do_hough_transform(stype='science+standard')

    # Solve for the pixel-to-wavelength solution
    red.fit(max_tries=2000, stype='science+standard', display=True)

    # Apply the wavelength calibration and display it
    red.apply_wavelength_calibration(wave_start=5000,
                                     wave_end=11000,
                                     wave_bin=1,
                                     stype='science+standard')

    red.load_standard(standard_name)
    red.compute_sensitivity()
    red.apply_flux_calibration()
    '''

    L745_fringe_count = L745_twodspec_flat.spectrum_list[0].count

    L745_fringe_continuum = lowess(L745_fringe_count,
                                np.arange(len(L745_fringe_count)),
                                frac=0.04,
                                return_sorted=False)
    L745_fringe_normalised = L745_fringe_count / L745_fringe_continuum

    L745_red_count = L745_twodspec_red.spectrum_list[0].count
    L745_red_continuum = lowess(L745_red_count,
                                np.arange(len(L745_red_count)),
                                frac=0.04,
                                return_sorted=False)
    L745_red_normalised = L745_red_count / L745_red_continuum
    L745_sed_correction = L745_fringe_continuum / L745_red_continuum
    L745_sed_correction /= np.nanmean(L745_sed_correction)

    L745_factor = (np.nanpercentile(L745_fringe_normalised[1000:1800], 95)) / (
        np.nanpercentile(L745_red_normalised[1000:1800], 5))
    L745_factor_mean = np.nanmean(L745_factor)

    L745_fringe_correction =\
        L745_fringe_normalised / L745_factor_mean *\
            L745_red_continuum * L745_sed_correction

    # Apply the flat correction
    L745_twodspec_red.spectrum_list[0].count -= L745_fringe_correction

    '''

    return red


def calibrate_blue(science, standard, standard_name):
    # Blue spectrum here
    blue = spectral_reduction.OneDSpec(log_level='INFO', log_file_name=None)

    blue.from_twodspec(standard, stype='standard')
    blue.from_twodspec(science, stype='science')

    blue.find_arc_lines(prominence=10, display=True, stype='science')
    blue.find_arc_lines(prominence=5, display=True, stype='standard')

    blue.initialise_calibrator(stype='science+standard')

    blue.add_user_atlas(elements=element_Hg_blue,
                        wavelengths=atlas_Hg_blue,
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
    blue.set_ransac_properties(filter_close=True, stype='science+standard')
    blue.do_hough_transform(stype='science+standard')

    # Solve for the pixel-to-wavelength solution
    blue.fit(max_tries=1000, stype='science+standard', display=True)

    # Apply the wavelength calibration and display it
    blue.apply_wavelength_calibration(wave_start=3000,
                                      wave_end=5800,
                                      wave_bin=1,
                                      stype='science+standard')

    blue.load_standard(standard_name)
    blue.compute_sensitivity()
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

L745_twodspec_red, L745_twodspec_blue, L745_twodspec_flat =\
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
        science_arc_fits,
        coeff_red=L745_twodspec_red.rec_coeff,
        coeff_blue=L745_twodspec_blue.rec_coeff)

standard_name = 'l74546a'

onedspec_blue = calibrate_blue(AT2019mtw_twodspec_blue, L745_twodspec_blue,
                               standard_name)
onedspec_red = calibrate_red(AT2019mtw_twodspec_red, L745_twodspec_red,
                             standard_name)

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

LCO_unit = 1e-19 * 0.8

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
plt.xlim(min(wave), max(wave))
plt.ylim(0, max(official_fits.data[0][0]) * LCO_unit * 1.05)
plt.xlabel('Wavelength / A')
plt.ylabel('Flux / (erg / s / cm / cm / A)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.title('AT 2019 MTW')

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
plt.xlim(min(wave_rest), max(wave_rest))
plt.ylim(0, max(official_fits.data[0][0]) * LCO_unit * 1.05)
plt.xlabel('Rest Wavelength / A')
plt.ylabel('Flux / (erg / s / cm / cm / A)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.title('AT 2019 MTW')

snex_fits = fits.open(
    '/Users/cylam/Downloads/AT2019mtw_20200321_redblu_181426.411.fits')[0]
wave_bin2 = float(snex_fits.header['CD1_1'])
wave_start2 = float(
    snex_fits.header['CRVAL1']) - float(snex_fits.header['CRPIX1']) * wave_bin2
spec_length2 = len(snex_fits.data[0][0])
wave2 = np.linspace(wave_start2, wave_start2 + wave_bin2 * spec_length2,
                    spec_length2)
plt.plot(wave2, snex_fits.data[0][0] * 30, color='red', label='SNEx')
