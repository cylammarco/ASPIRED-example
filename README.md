# ASPIRED-example
Example reduction with ASPIRED

## Use cases

### Spectrum - full reduction
1. [x] Dataset with science and standard field-flattened images and the respective arc image.

### Spectrum - ADU spectrum extraction only (No flux calibration)
2. [x] Dataset with science field-flattened image and the arc image only.

### Spectrum - wavelength calibration only (Pre wavelength-calibrated)
3. [x] Dataset with science and standard field-flattened images only.

### Spectrum - other cases for full or partial reduction
4. [x] User supplied trace(s) for light spectrum extraction (only to extract ADU/s of the spectra).
5. [x] User supplied trace(s) for arc extraction (only to get wavelength calibration polynomial).
6. [x] User supplied wavelength calibration polynomial coefficients.
7. [x] User supplied line list.
8. [x] User supplied pixel-to-wavelength mapping (not fitted).
9. [x] User supplied sensitivity curve.
10. [x] Flux calibration for user supplied wavelength calibrated science and standard 1D spectra.

## SPRAT - Liverpool Telescope

A jupyter notebook showing a full reduction of a SPRAT spectrum of the M dwarf [LHS6328](http://simbad.u-strasbg.fr/simbad/sim-id?Ident=LHS++6328), calibrated against the standard star [Hiltner 102](http://www.ing.iac.es/Astronomy/observing/manuals/html_manuals/tech_notes/tn065-100/h102.html) and with a Xe arc lamp.

## ISIS - William Herschel Telescope

A jupyter notebook showing a full reduction of the red arm of ISIS of [PSO J180.1536+62.5419](http://simbad.u-strasbg.fr/simbad/sim-id?Ident=%4015306489&Name=PSO%20J180.1536%2b62.5419&submit=submit), calibrated against [G93-48](https://www.eso.org/sci/observing/tools/standards/spectra/g93_48.html) and with a CuAr+CuNe arc lamp.

## DOLORES - Telescopio Nazionale Galileo

A plain Python script to show a full reduction of the red and blue arm of [ZTF19aamsetj](https://www.wis-tns.org/object/2019cad), calibrated against [HD 19445](http://www.ing.iac.es/Astronomy/observing/manuals/html_manuals/tech_notes/tn065-100/hd194.html) and with a Ar+Kr+Ne+Hg arc lamp.

## OSIRIS - Gran Telescopio Canarias

A plain Python script to show a full reduction of the R1000B and R2500U spectra of ZTF-BLAP-01, calibrated with a Hg+Ar+Ne lamp and Hg+Ar+Xe lamp, respectively. The flux calibration was performed with [Feige 110](http://www.ing.iac.es/Astronomy/observing/manuals/html_manuals/tech_notes/tn065-100/f110.html). Both interstellar and atmospheric extinctions are corrected.

## FLOYDS - Faulkes South Telescope

A plain Python script to show a full reduction of the red and blue FLOYDS spectra of [AT2019MTW](https://www.wis-tns.org/object/2019mtw), calibrated against [L 745-46A](http://www.ing.iac.es/Astronomy/observing/manuals/html_manuals/tech_notes/tn065-100/l745.html). 2D distortion correction is applied to straighten the spectra; the 1D fringe pattern is subtracted from the red spectrum; and the Telluric absorptions are corrected.
