# VAST Cross-Matching Utility

This package provides Python routines and a CLI for cross-matching different epochs of VAST catalogues of the same field. The median astrometric offsets and the peak flux multiplicative and additive offsets are calculated and stored in a CSV file and/or a SQLite database.

## Installation

Install with pip, e.g.

```text
pip install git+https://github.com/marxide/vast-xmatch.git
```

## CLI Usage

A script named `vast_xmatch` should now be available in your PATH. Run it with the `--help` option to see a list of available commands.

```text
$ vast_xmatch --help
Usage: vast_xmatch [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  export  Export database corrections to CSV.
  qc      Crossmatch and output positional and flux corrections.
```

### `vast_xmatch qc`

The `qc` command will cross-match the sources from two VAST catalogs using a simple nearest-neighbor algorithm with an optional maximum separation limit. The first input catalog is designated the reference catalog where all position and flux values are considered "the truth".

Positional corrections in RA and Dec (∆RA and ∆Dec) are calculated as the median positional offset between cross-matched sources. The median absolute deviation from the median (MADFM) of the positional offsets is also calculated. The returned values, optionally written to a CSV file and/or SQLite database, are intended to be used to correct the input catalog to match the input reference catalog:

* `RA' = RA + ∆RA / cos(Dec)`
* `Dec' = Dec + ∆Dec`

Where `RA` is the RA of the input catalog sources, `∆RA` is the RA correction offset written out to the CSV/database and `RA'` is the corrected RA of the input catalog sources such that they agree (on average) with the input reference catalog. Likewise for Dec.

Flux corrections are calculated by fitting the input catalog peak fluxes to a straight line. The function `y = mx + b` is fit to the reference peak fluxes vs the peak fluxes using orthogonal distance regression with `scipy.odr`. The gradient and offset of the fit (`m` and `b`, respectively) are converted to multiplicative and additive corrections that can be used to correct the input catalog to match the input reference catalog:

* `S' = Cm * (S + Ca)`

Where `S` is the input catalog source peak fluxes, `Cm` is the multiplicative correction, `Ca` is the additive correction, and `S'` is the corrected source peak flux. The gradient or offset of the flux fitting can also be fixed with the `--fix-m` and `--fix-b`.

**WARNING:** The `qc` command will print the fitted parameter values to the log for debugging purposes. These values should not be used for the corrections described above. They are converted to appropriate correction values to suit the equations given above before being written to a CSV file or database. Always use the values in the CSV file or database for corrections, never the values in the logs.

**WARNING:** Be aware of the units when applying the corrections. See the `--help` output.

Run with the `--help` option to see full usage information.

```text
$ vast_xmatch qc --help
Usage: vast_xmatch qc [OPTIONS] REFERENCE_CATALOG_PATH CATALOG_PATH

  Crossmatch a catalog with a reference catalog and output the positional
  and flux corrections for the input catalog.

Options:
  --radius ANGLE_QUANTITY       Maximum separation limit for nearest-neighbour
                                crossmatch. Accepts any string understood by
                                astropy.coordinates.Angle.

  --condon                      Calculate Condon (1997) flux errors and use
                                them instead of the original errors. Will also
                                correct the peak flux values for noise.
                                Requires that the input catalogs follow the
                                VAST naming convention, e.g. for COMBINED
                                images:
                                VAST_0102-06A.EPOCH01.I.selavy.components.txt,
                                and for TILE images:
                                EPOCH01/TILES/STOKESI_SELAVY/selavy-image.i.SB
                                9667.cont.VAST_0102-06A.linmos.taylor.0.restor
                                ed.components.txt. Note that for TILE images,
                                the epoch is determined from the full path. If
                                the input catalogs do not follow this
                                convention, then the PSF sizes must be
                                supplied using --psf-reference and/or --psf.
                                The deafult behaviour is to lookup the PSF
                                sizes from the VAST metadata server.

  --psf-reference FLOAT...      If using --condon and not using --lookup-psf,
                                use this specified PSF size in arcsec for
                                `reference_catalog`.

  --psf FLOAT...                If using --condon and not using --lookup-psf,
                                use this specified PSF size in arcsec for
                                `catalog`.

  --fix-m                       Fix the gradient to 1.0 when fitting.
  --fix-b                       Fix the offset to 0.0 when fitting.
  -v, --verbose
  --aegean                      Input catalog is an Aegean CSV.
  --positional-unit ANGLE_UNIT  Positional correction output unit. Must be an
                                angular unit. Default is arcsec.

  --flux-unit FLUX_UNIT         Flux correction output unit. Must be a
                                spectral flux density unit. Do not include a
                                beam divisor, this will be automatically added
                                for peak flux values. Default is mJy.

  --csv-output FILE             Path to write CSV of positional and flux
                                corrections. Only available if `catalog`
                                follows VAST naming conventions as the field
                                and epoch must be known. If the file exists,
                                the corrections will be appended. Corrections
                                are written in the units specified by
                                --positional-unit and --flux-unit. To apply
                                the corrections, use the following equations:
                                ra corrected = ra + ra_correction / cos(dec);
                                dec corrected = dec + dec_correction; flux
                                peak corrected =
                                flux_peak_correction_multiplicative * (flux
                                peak + flux_peak_correction_additive. Note
                                that these correction values have been
                                modified to suit these equations and are
                                different from the fitted values shown in the
                                logs.

  --sqlite-output FILE          Write corrections to the given SQLite3
                                database. Will create the database if it
                                doesn't exist and replace existing corrections
                                for matching catalogs. See the help for --csv-
                                output for more information on the correction
                                values. However, unlike the CSV output, the
                                positional unit for the database is always
                                degrees and the flux unit is always Jy[/beam].

  --plot-path PATH              Save plots of the crossmatched sources
                                positional offsets and flux ratios as a PNG
                                image to the given directory. If the directory
                                does not exist, it will be created. The axis
                                units are specified by --positional-unit and
                                --flux-unit. The output filenames will be the
                                name of the input catalog with the suffix
                                _positional_offset.png and _flux_ratio.png.

  --plot-pos-gridsize INTEGER   Number of hexagons in the x-direction of the
                                positional offset plot. Default is 50.

  --help                        Show this message and exit.
```

### `vast_xmatch export`

Run with the `--help` option to see full usage information.

```text
$ vast_xmatch export --help
Usage: vast_xmatch export [OPTIONS] DATABASE_PATH CSV_PATH

  Export the contents of a SQLite database of VAST corrections to a CSV
  file.

Options:
  --vast-type [TILE|COMBINED]
  --positional-unit ANGLE_UNIT  Positional correction output unit. Must be an
                                angular unit. Default is arcsec.

  --flux-unit FLUX_UNIT         Flux correction output unit. Must be a
                                spectral flux density unit. Do not include a
                                beam divisor, this will be automatically added
                                for peak flux values. Default is mJy.

  -v, --verbose
  --help                        Show this message and exit.
```
