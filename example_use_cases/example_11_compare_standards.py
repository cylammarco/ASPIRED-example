from aspired import spectral_reduction
from matplotlib import pyplot as plt

plt.figure(1, figsize=(8, 6))
plt.clf()

onedspec_ing_mas = spectral_reduction.OneDSpec()
onedspec_ing_mas.load_standard(target='f110', library='ing_mas')
plt.plot(onedspec_ing_mas.fluxcal.standard_wave_true,
         onedspec_ing_mas.fluxcal.standard_fluxmag_true,
         label='ING mas')

onedspec_ing_oke = spectral_reduction.OneDSpec()
onedspec_ing_oke.load_standard(target='f110', library='ing_oke')
plt.plot(onedspec_ing_oke.fluxcal.standard_wave_true,
         onedspec_ing_oke.fluxcal.standard_fluxmag_true,
         label='ING oke')

onedspec_ing_sto = spectral_reduction.OneDSpec()
onedspec_ing_sto.load_standard(target='f110', library='ing_sto')
plt.plot(onedspec_ing_sto.fluxcal.standard_wave_true,
         onedspec_ing_sto.fluxcal.standard_fluxmag_true,
         label='ING sto')

onedspec_eso = spectral_reduction.OneDSpec()
onedspec_eso.load_standard(target='Feige110', library='esoxshooter')
plt.plot(onedspec_eso.fluxcal.standard_wave_true,
         onedspec_eso.fluxcal.standard_fluxmag_true,
         label='ESO')

plt.xlim(3000, 10000)
plt.ylim(7e-15, 6e-13)

plt.xlabel(r'Wavelength ($\mathrm{\AA}$)')
plt.ylabel(r'Flux (erg cm$^{-2}$ s$^{-1} \mathrm{\AA} ^{-1}$)')
plt.grid()
plt.tight_layout()
plt.legend()
plt.show()
