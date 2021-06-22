import numpy as np
from aspired import spectral_reduction
from aspired import image_reduction
from spectres import spectres
from matplotlib import pyplot as plt

spatial_mask = np.arange(500, 1500)

# Science frame
ztf19aamsetj_frame_lrr1 = image_reduction.ImageReduction(
    'ZTF19aamsetj_LRR1.list',
    cosmicray=True,
    psfmodel='gaussyx',
    log_level='INFO',
    log_file_name='None')
ztf19aamsetj_frame_lrr2 = image_reduction.ImageReduction(
    'ZTF19aamsetj_LRR2.list',
    cosmicray=True,
    psfmodel='gaussyx',
    log_level='INFO',
    log_file_name='None')
ztf19aamsetj_frame_lrb1 = image_reduction.ImageReduction(
    'ZTF19aamsetj_LRB1.list',
    cosmicray=True,
    psfmodel='gaussyx',
    log_level='INFO',
    log_file_name='None')
ztf19aamsetj_frame_lrb2 = image_reduction.ImageReduction(
    'ZTF19aamsetj_LRB2.list',
    cosmicray=True,
    psfmodel='gaussyx',
    log_level='INFO',
    log_file_name='None')

ztf19aamsetj_frame_lrr1.reduce()
ztf19aamsetj_frame_lrr2.reduce()
ztf19aamsetj_frame_lrb1.reduce()
ztf19aamsetj_frame_lrb2.reduce()

ztf19aamsetj_twodspec_lrr1 = spectral_reduction.TwoDSpec(
    ztf19aamsetj_frame_lrr1,
    spatial_mask=spatial_mask,
    cosmicray=True,
    log_level='INFO',
    log_file_name='None')
ztf19aamsetj_twodspec_lrr2 = spectral_reduction.TwoDSpec(
    ztf19aamsetj_frame_lrr2,
    spatial_mask=spatial_mask,
    cosmicray=True,
    log_level='INFO',
    log_file_name='None')
ztf19aamsetj_twodspec_lrb1 = spectral_reduction.TwoDSpec(
    ztf19aamsetj_frame_lrb1,
    spatial_mask=spatial_mask,
    cosmicray=True,
    log_level='INFO',
    log_file_name='None')
ztf19aamsetj_twodspec_lrb2 = spectral_reduction.TwoDSpec(
    ztf19aamsetj_frame_lrb2,
    spatial_mask=spatial_mask,
    cosmicray=True,
    log_level='INFO',
    log_file_name='None')

ztf19aamsetj_twodspec_lrr1.ap_trace(nspec=1,
                                    nwindow=10,
                                    fit_deg=1,
                                    percentile=2,
                                    display=True)
ztf19aamsetj_twodspec_lrr1.ap_extract(display=True)

ztf19aamsetj_twodspec_lrr2.ap_trace(nspec=1,
                                    nwindow=10,
                                    fit_deg=1,
                                    percentile=2,
                                    display=True)
ztf19aamsetj_twodspec_lrr2.ap_extract(display=True)

ztf19aamsetj_twodspec_lrb1.ap_trace(nspec=1, fit_deg=2, display=True)
ztf19aamsetj_twodspec_lrb1.ap_extract(display=True)

ztf19aamsetj_twodspec_lrb2.ap_trace(nspec=1, fit_deg=2, display=True)
ztf19aamsetj_twodspec_lrb2.ap_extract(display=True)

ztf19aamsetj_twodspec_lrb1.apply_twodspec_mask_to_arc()
ztf19aamsetj_twodspec_lrb2.apply_twodspec_mask_to_arc()
ztf19aamsetj_twodspec_lrr1.apply_twodspec_mask_to_arc()
ztf19aamsetj_twodspec_lrr2.apply_twodspec_mask_to_arc()

ztf19aamsetj_twodspec_lrb1.extract_arc_spec(display=False)
ztf19aamsetj_twodspec_lrb2.extract_arc_spec(display=False)
ztf19aamsetj_twodspec_lrr1.extract_arc_spec(display=False)
ztf19aamsetj_twodspec_lrr2.extract_arc_spec(display=False)

# Standard frames
hd93521_frame_lrr = image_reduction.ImageReduction('HD93521_LRR.list',
                                                   cosmicray=False,
                                                   psfmodel='gauss',
                                                   log_level='INFO',
                                                   log_file_name='None')
hd93521_frame_lrb = image_reduction.ImageReduction('HD93521_LRB.list',
                                                   cosmicray=False,
                                                   psfmodel='gauss',
                                                   log_level='INFO',
                                                   log_file_name='None')

hd93521_frame_lrr.reduce()
hd93521_frame_lrb.reduce()

hd93521_twodspec_lrr = spectral_reduction.TwoDSpec(hd93521_frame_lrr,
                                                   spatial_mask=spatial_mask,
                                                   cosmicray=True,
                                                   log_level='INFO',
                                                   log_file_name='None')
hd93521_twodspec_lrb = spectral_reduction.TwoDSpec(hd93521_frame_lrb,
                                                   spatial_mask=spatial_mask,
                                                   cosmicray=True,
                                                   log_level='INFO',
                                                   log_file_name='None')

hd93521_twodspec_lrb.ap_trace(nspec=1, display=True)
hd93521_twodspec_lrb.ap_extract(apwidth=15, display=True)

hd93521_twodspec_lrr.ap_trace(nspec=1, display=True)
hd93521_twodspec_lrr.ap_extract(apwidth=15, display=True)

hd93521_twodspec_lrb.apply_twodspec_mask_to_arc()
hd93521_twodspec_lrr.apply_twodspec_mask_to_arc()

hd93521_twodspec_lrb.extract_arc_spec(display=True)
hd93521_twodspec_lrr.extract_arc_spec(display=True)

# Handle 1D Science spectrum
ztf19aamsetj_onedspec_lrb1 = spectral_reduction.OneDSpec(log_level='INFO',
                                                         log_file_name=None)
ztf19aamsetj_onedspec_lrb1.from_twodspec(ztf19aamsetj_twodspec_lrb1,
                                         stype='science')
ztf19aamsetj_onedspec_lrb1.from_twodspec(hd93521_twodspec_lrb,
                                         stype='standard')

ztf19aamsetj_onedspec_lrb2 = spectral_reduction.OneDSpec(log_level='INFO',
                                                         log_file_name=None)
ztf19aamsetj_onedspec_lrb2.from_twodspec(ztf19aamsetj_twodspec_lrb2,
                                         stype='science')
ztf19aamsetj_onedspec_lrb2.from_twodspec(hd93521_twodspec_lrb,
                                         stype='standard')

ztf19aamsetj_onedspec_lrr1 = spectral_reduction.OneDSpec(log_level='INFO',
                                                         log_file_name=None)
ztf19aamsetj_onedspec_lrr1.from_twodspec(ztf19aamsetj_twodspec_lrr1,
                                         stype='science')
ztf19aamsetj_onedspec_lrr1.from_twodspec(hd93521_twodspec_lrr,
                                         stype='standard')

ztf19aamsetj_onedspec_lrr2 = spectral_reduction.OneDSpec(log_level='INFO',
                                                         log_file_name=None)
ztf19aamsetj_onedspec_lrr2.from_twodspec(ztf19aamsetj_twodspec_lrr2,
                                         stype='science')
ztf19aamsetj_onedspec_lrr2.from_twodspec(hd93521_twodspec_lrr,
                                         stype='standard')

# Extract arc spectrum
ztf19aamsetj_onedspec_lrb1.find_arc_lines(prominence=50.,
                                          stype='science+standard')
ztf19aamsetj_onedspec_lrb2.find_arc_lines(prominence=50.,
                                          stype='science+standard')
ztf19aamsetj_onedspec_lrr1.find_arc_lines(prominence=100.,
                                          stype='science+standard')
ztf19aamsetj_onedspec_lrr2.find_arc_lines(prominence=100.,
                                          stype='science+standard')

# Configure the wavelength calibrator
ztf19aamsetj_onedspec_lrb1.initialise_calibrator(stype='science+standard')
ztf19aamsetj_onedspec_lrb2.initialise_calibrator(stype='science+standard')
ztf19aamsetj_onedspec_lrr1.initialise_calibrator(stype='science+standard')
ztf19aamsetj_onedspec_lrr2.initialise_calibrator(stype='science+standard')

ztf19aamsetj_onedspec_lrb1.set_hough_properties(num_slopes=1000,
                                                xbins=100,
                                                ybins=100,
                                                min_wavelength=3000,
                                                max_wavelength=8500,
                                                stype='science+standard')
ztf19aamsetj_onedspec_lrb2.set_hough_properties(num_slopes=1000,
                                                xbins=100,
                                                ybins=100,
                                                min_wavelength=3000,
                                                max_wavelength=8500,
                                                stype='science+standard')
ztf19aamsetj_onedspec_lrr1.set_hough_properties(num_slopes=2000,
                                                xbins=200,
                                                ybins=200,
                                                min_wavelength=4500,
                                                max_wavelength=10500,
                                                stype='science+standard')
ztf19aamsetj_onedspec_lrr2.set_hough_properties(num_slopes=2000,
                                                xbins=200,
                                                ybins=200,
                                                min_wavelength=4500,
                                                max_wavelength=10500,
                                                stype='science+standard')

ztf19aamsetj_onedspec_lrb1.set_ransac_properties(filter_close=True,
                                                 stype='science+standard')
ztf19aamsetj_onedspec_lrb2.set_ransac_properties(filter_close=True,
                                                 stype='science+standard')
ztf19aamsetj_onedspec_lrr1.set_ransac_properties(filter_close=True,
                                                 stype='science+standard')
ztf19aamsetj_onedspec_lrr1.set_ransac_properties(filter_close=True,
                                                 stype='science+standard')

ztf19aamsetj_onedspec_lrb1.add_atlas(elements=['He'], stype='science+standard')
ztf19aamsetj_onedspec_lrb2.add_atlas(elements=['He'], stype='science+standard')

atlas = [
    5769.598, 5820.1558, 5852.4879, 5944.8342, 5975.5340, 6029.9969, 6074.3377,
    6096.1631, 6143.0626, 6163.5939, 6217.2812, 6266.4950, 6334.4278,
    6382.9917, 6506.5281, 6532.8822, 6598.9529, 6678.2762, 6717.0430,
    6929.4673, 7173.9381, 7245.1666, 7272.9360, 7383.9800, 7435.7800,
    7488.8712, 7587.4136, 7601.5457, 7948.176, 8059.5048, 8112.9012, 8136.4054,
    8263.2426, 8298.1099, 8495.3598, 8521.4420, 8654.3831, 8704.1116,
    8776.7505, 8928.6934, 9122.9670, 9657.7860
]
elements = ['ArKrNeHg'] * len(atlas)

ztf19aamsetj_onedspec_lrr1.load_user_atlas(elements=elements,
                                           wavelengths=atlas,
                                           stype='science+standard')
ztf19aamsetj_onedspec_lrr2.load_user_atlas(elements=elements,
                                           wavelengths=atlas,
                                           stype='science+standard')

ztf19aamsetj_onedspec_lrb1.do_hough_transform()
ztf19aamsetj_onedspec_lrb2.do_hough_transform()
ztf19aamsetj_onedspec_lrr1.do_hough_transform()
ztf19aamsetj_onedspec_lrr2.do_hough_transform()

# Solve for the pixel-to-wavelength solution
ztf19aamsetj_onedspec_lrb1.fit(max_tries=1000,
                               stype='science+standard',
                               display=False)
ztf19aamsetj_onedspec_lrb2.fit(max_tries=1000,
                               stype='science+standard',
                               display=False)
ztf19aamsetj_onedspec_lrr1.fit(max_tries=500,
                               stype='science+standard',
                               display=False)
ztf19aamsetj_onedspec_lrr2.fit(max_tries=500,
                               stype='science+standard',
                               display=False)

# Apply the wavelength calibration and display it
ztf19aamsetj_onedspec_lrb1.apply_wavelength_calibration(
    stype='science+standard')
ztf19aamsetj_onedspec_lrb2.apply_wavelength_calibration(
    stype='science+standard')
ztf19aamsetj_onedspec_lrr1.apply_wavelength_calibration(
    stype='science+standard')
ztf19aamsetj_onedspec_lrr2.apply_wavelength_calibration(
    stype='science+standard')

# Get the standard from the library
ztf19aamsetj_onedspec_lrb1.load_standard(target='hd93521')
ztf19aamsetj_onedspec_lrb2.load_standard(target='hd93521')
ztf19aamsetj_onedspec_lrr1.load_standard(target='hd93521')
ztf19aamsetj_onedspec_lrr2.load_standard(target='hd93521')

ztf19aamsetj_onedspec_lrb1.compute_sensitivity(k=3,
                                               method='interpolate',
                                               mask_fit_size=1,
                                               extinction_correction=True)
ztf19aamsetj_onedspec_lrb2.compute_sensitivity(k=3,
                                               method='interpolate',
                                               mask_fit_size=1,
                                               extinction_correction=True)
ztf19aamsetj_onedspec_lrr1.compute_sensitivity(k=3,
                                               method='interpolate',
                                               mask_fit_size=1,
                                               extinction_correction=True)
ztf19aamsetj_onedspec_lrr2.compute_sensitivity(k=3,
                                               method='interpolate',
                                               mask_fit_size=1,
                                               extinction_correction=True)

ztf19aamsetj_onedspec_lrb1.apply_flux_calibration(stype='science+standard')
ztf19aamsetj_onedspec_lrb2.apply_flux_calibration(stype='science+standard')
ztf19aamsetj_onedspec_lrr1.apply_flux_calibration(stype='science+standard')
ztf19aamsetj_onedspec_lrr2.apply_flux_calibration(stype='science+standard')

# Apply atmospheric extinction correction
ztf19aamsetj_onedspec_lrb1.set_atmospheric_extinction(location='orm')
ztf19aamsetj_onedspec_lrb2.set_atmospheric_extinction(location='orm')
ztf19aamsetj_onedspec_lrr1.set_atmospheric_extinction(location='orm')
ztf19aamsetj_onedspec_lrr2.set_atmospheric_extinction(location='orm')

ztf19aamsetj_onedspec_lrb1.apply_atmospheric_extinction_correction()
ztf19aamsetj_onedspec_lrb2.apply_atmospheric_extinction_correction()
ztf19aamsetj_onedspec_lrr1.apply_atmospheric_extinction_correction()
ztf19aamsetj_onedspec_lrr2.apply_atmospheric_extinction_correction()

ztf19aamsetj_onedspec_lrb1.inspect_reduced_spectrum(wave_min=3000,
                                                    wave_max=8500)
ztf19aamsetj_onedspec_lrb2.inspect_reduced_spectrum(wave_min=3000,
                                                    wave_max=8500)
ztf19aamsetj_onedspec_lrr1.inspect_reduced_spectrum(wave_min=4500,
                                                    wave_max=10100)
ztf19aamsetj_onedspec_lrr2.inspect_reduced_spectrum(wave_min=4500,
                                                    wave_max=10100)

ztf19aamsetj_onedspec_lrb1.save_csv(output='wavelength+flux',
                                    filename='ztf19aamsetj_lrb1')
ztf19aamsetj_onedspec_lrb2.save_csv(output='wavelength+flux',
                                    filename='ztf19aamsetj_lrb2')
ztf19aamsetj_onedspec_lrr1.save_csv(output='wavelength+flux',
                                    filename='ztf19aamsetj_lrr1')
ztf19aamsetj_onedspec_lrr2.save_csv(output='wavelength+flux',
                                    filename='ztf19aamsetj_lrr2')

wave1 = ztf19aamsetj_onedspec_lrb1.science_spectrum_list[0].wave
flux1 = ztf19aamsetj_onedspec_lrb1.science_spectrum_list[0].flux
flux_err1 = ztf19aamsetj_onedspec_lrb1.science_spectrum_list[0].flux_err

wave2 = ztf19aamsetj_onedspec_lrb2.science_spectrum_list[0].wave
flux2 = ztf19aamsetj_onedspec_lrb2.science_spectrum_list[0].flux
flux_err2 = ztf19aamsetj_onedspec_lrb2.science_spectrum_list[0].flux_err

flux2_resampled, flux_err2_resampled = spectres(wave1, wave2, flux2, flux_err2)

wave3 = ztf19aamsetj_onedspec_lrr1.science_spectrum_list[0].wave
flux3 = ztf19aamsetj_onedspec_lrr1.science_spectrum_list[0].flux
flux_err3 = ztf19aamsetj_onedspec_lrr1.science_spectrum_list[0].flux_err

wave4 = ztf19aamsetj_onedspec_lrr2.science_spectrum_list[0].wave
flux4 = ztf19aamsetj_onedspec_lrr2.science_spectrum_list[0].flux
flux_err4 = ztf19aamsetj_onedspec_lrr2.science_spectrum_list[0].flux_err

flux4_resampled, flux_err4_resampled = spectres(wave3, wave4, flux4, flux_err4)

plt.figure(1, figsize=(12, 6))
plt.clf()
plt.plot(wave1, flux1)
plt.plot(wave2, flux2)
plt.plot(wave3, flux3)
plt.plot(wave4, flux4)
plt.ylim(0, 1e-16)
plt.grid()
plt.xlabel('Wavelength / A')
plt.ylabel('Flux')
plt.tight_layout()
plt.savefig('ztf19aamsetj_four_independently_calibrated_spectra.png')

plt.figure(2, figsize=(12, 6))
plt.clf()
plt.plot(wave1, (flux1 / flux_err1 + flux2_resampled / flux_err2_resampled) /
         (1 / flux_err1 + 1 / flux_err2_resampled))
plt.plot(
    wave3, 2 * (flux3 / flux_err3 + flux4_resampled / flux_err4_resampled) /
    (1 / flux_err3 + 1 / flux_err4_resampled))
plt.ylim(0, 1e-16)
plt.grid()
plt.xlabel('Wavelength / A')
plt.ylabel('Flux')
plt.tight_layout()
plt.savefig('ztf19aamsetj_LRR_LRB_weighted_mean_spectra.png')

np.savetxt('LRB_weighted_mean.txt', 
    np.column_stack(
        (wave1, (flux1 / flux_err1 + flux2_resampled / flux_err2_resampled) /
         (1 / flux_err1 + 1 / flux_err2_resampled))))
np.savetxt('LRR_weighted_mean.txt',
    np.column_stack(
        (wave3, (flux3 / flux_err3 + flux4_resampled / flux_err4_resampled) /
         (1 / flux_err3 + 1 / flux_err4_resampled))))

wave_coadd = np.arange(min(wave1), max(wave3),
                       np.median(np.ediff1d(np.concatenate((wave1, wave3)))))
flux1_coadd, flux_err1_coadd = spectres(wave_coadd, wave1, flux1, flux_err1)
flux2_coadd, flux_err2_coadd = spectres(wave_coadd, wave2, flux2, flux_err2)
flux3_coadd, flux_err3_coadd = spectres(wave_coadd, wave3, flux3, flux_err3)
flux4_coadd, flux_err4_coadd = spectres(wave_coadd, wave4, flux4, flux_err4)

flux3_coadd *= 2
flux4_coadd *= 2

masked_flux1_coadd = np.ma.masked_array(flux1_coadd, ~np.isfinite(flux1_coadd))
masked_flux2_coadd = np.ma.masked_array(flux2_coadd, ~np.isfinite(flux2_coadd))
masked_flux3_coadd = np.ma.masked_array(flux3_coadd, ~np.isfinite(flux3_coadd))
masked_flux4_coadd = np.ma.masked_array(flux4_coadd, ~np.isfinite(flux4_coadd))

flux_err1_coadd[~np.isfinite(flux_err1_coadd)] = np.inf
flux_err2_coadd[~np.isfinite(flux_err2_coadd)] = np.inf
flux_err3_coadd[~np.isfinite(flux_err3_coadd)] = np.inf
flux_err4_coadd[~np.isfinite(flux_err4_coadd)] = np.inf

flux_coadd = np.ma.average(
    (masked_flux1_coadd, masked_flux2_coadd, masked_flux3_coadd,
     masked_flux4_coadd),
    axis=0,
    weights=(1. / flux_err1_coadd, 1. / flux_err2_coadd, 1. / flux_err3_coadd,
             1. / flux_err4_coadd))

wave_coadd_bin2 = wave_coadd[::2]
flux_coadd_bin2 = spectres(wave_coadd_bin2, wave_coadd, flux_coadd)

plt.figure(3, figsize=(12, 6))
plt.clf()
plt.plot(wave_coadd, flux_coadd, label='Weighted Mean')
plt.plot(wave_coadd_bin2,
         flux_coadd_bin2 - 2e-17,
         label='Weighted Mean (bin 2, shifted by 2E-17)')
plt.ylim(0, 1e-16)
plt.grid()
plt.xlabel('Wavelength / A')
plt.ylabel('Flux')
plt.legend()
plt.tight_layout()
plt.savefig('ztf19aamsetj_total_weighted_mean_spectra.png')

np.savetxt('coadd_spectrum.txt', np.column_stack((wave_coadd, flux_coadd)))
np.savetxt('coadd_spectrum_bin2.txt', np.column_stack((wave_coadd_bin2, flux_coadd_bin2))
           )
