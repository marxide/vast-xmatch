import logging
from pathlib import Path
from typing import Optional, Tuple

from astropy.coordinates import Angle
import astropy.units as u
import click
import pandas as pd
from uncertainties import ufloat

from vast_xmatch.catalogs import Catalog, UnknownFilenameConvention
from vast_xmatch.crossmatch import (
    crossmatch_qtables,
    calculate_positional_offsets,
    calculate_flux_offsets,
)
import vast_xmatch.db as db
from vast_xmatch.plot import positional_offset_plot, flux_ratio_plot


class _AstropyUnitType(click.ParamType):
    def convert(self, value, param, ctx, unit_physical_type):
        try:
            unit = u.Unit(value)
        except ValueError:
            self.fail(f"astropy.units.Unit does not understand: {value}.")
        if unit.physical_type != unit_physical_type:
            self.fail(
                f"{unit} is a {unit.physical_type} unit. It must be of type"
                f" {unit_physical_type}."
            )
        else:
            return unit


class AngleUnitType(_AstropyUnitType):
    name = "angle_unit"

    def convert(self, value, param, ctx):
        return super().convert(value, param, ctx, "angle")


class FluxUnitType(_AstropyUnitType):
    name = "flux_unit"

    def convert(self, value, param, ctx):
        return super().convert(value, param, ctx, "spectral flux density")


class AngleQuantityType(click.ParamType):
    name = "angle_quantity"

    def convert(self, value, param, ctx):
        try:
            angle = Angle(value)
            return angle
        except ValueError:
            self.fail(f"astropy.coordinates.Angle does not understand: {value}.")


ANGLE_UNIT_TYPE = AngleUnitType()
FLUX_UNIT_TYPE = FluxUnitType()
ANGLE_QUANTITY_TYPE = AngleQuantityType()
logger = logging.getLogger("vast_xmatch")


def transform_epoch_raw(epoch: str) -> str:
    suffix = "x" if epoch.endswith("x") else ""
    epoch_num = int(epoch.replace("EPOCH", "").rstrip("x"))
    return f"{epoch_num}{suffix}"


def _transform_epoch_vastp(epoch: str) -> str:
    return f"vastp{transform_epoch_raw(epoch)}"


def _default_none(ctx, param, value):
    if len(value) == 0:
        return None
    else:
        return value


@click.group()
def cli():
    pass


@cli.command(
    name="qc",
    help=(
        "Crossmatch a catalog with a reference catalog and output the positional and"
        " flux corrections for the input catalog."
    ),
    short_help="Crossmatch and output positional and flux corrections.",
)
@click.argument("reference_catalog_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("catalog_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--radius",
    type=ANGLE_QUANTITY_TYPE,
    default="10 arcsec",
    help=(
        "Maximum separation limit for nearest-neighbour crossmatch. Accepts any "
        "string understood by astropy.coordinates.Angle."
    ),
)
@click.option(
    "--condon",
    is_flag=True,
    help=(
        "Calculate Condon (1997) flux errors and use them instead of the original "
        "errors. Will also correct the peak flux values for noise. Requires that the "
        "input catalogs follow the VAST naming convention, e.g. for COMBINED images: "
        "VAST_0102-06A.EPOCH01.I.selavy.components.txt, and for TILE images: EPOCH01/"
        "TILES/STOKESI_SELAVY/selavy-image.i.SB9667.cont.VAST_0102-06A.linmos.taylor.0"
        ".restored.components.txt. Note that for TILE images, the epoch is determined "
        "from the full path. If the input catalogs do not follow this convention, then "
        "the PSF sizes must be supplied using --psf-reference and/or --psf."
    ),
)
@click.option(
    "--askap-database",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help=(
        "Path to a local copy of the ASKAP surveys database repository. Required if"
        " using --condon without also specifying the PSF sizes with --psf."
    ),
)
@click.option(
    "--psf-reference",
    nargs=2,
    type=float,
    callback=_default_none,
    help=(
        "If using --condon, use this specified PSF size in arcsec for"
        " `reference_catalog`."
    ),
)
@click.option(
    "--psf",
    nargs=2,
    type=float,
    callback=_default_none,
    help=(
        "If using --condon and not using --askap-database, use this specified PSF size"
        " in arcsec for `catalog`."
    ),
)
@click.option("--fix-m", is_flag=True, help="Fix the gradient to 1.0 when fitting.")
@click.option("--fix-b", is_flag=True, help="Fix the offset to 0.0 when fitting.")
@click.option("-v", "--verbose", is_flag=True)
@click.option("--aegean", is_flag=True, help="Input catalog is an Aegean CSV.")
@click.option(
    "--positional-unit",
    type=ANGLE_UNIT_TYPE,
    default="arcsec",
    help=(
        "Positional correction output unit. Must be an angular unit. Default is arcsec."
    ),
)
@click.option(
    "--flux-unit",
    type=FLUX_UNIT_TYPE,
    default="mJy",
    help=(
        "Flux correction output unit. Must be a spectral flux density unit. Do not"
        " include a beam divisor, this will be automatically added for peak flux"
        " values. Default is mJy."
    ),
)
@click.option(
    "--csv-output",
    type=click.Path(dir_okay=False, writable=True),
    help=(
        "Path to write CSV of positional and flux corrections. Only available if"
        " `catalog` follows VAST naming conventions as the field and epoch must be"
        " known. If the file exists, the corrections will be appended. Corrections are"
        " written in the units specified by --positional-unit and --flux-unit. To apply"
        " the corrections, use the following equations: ra corrected = ra +"
        " ra_correction / cos(dec); dec corrected = dec + dec_correction; flux peak"
        " corrected = flux_peak_correction_multiplicative * (flux peak +"
        " flux_peak_correction_additive. Note that these correction values have been"
        " modified to suit these equations and are different from the fitted values"
        " shown in the logs."
    ),
)
@click.option(
    "--sqlite-output",
    type=click.Path(dir_okay=False, file_okay=True),
    help=(
        "Write corrections to the given SQLite3 database. Will create the database if"
        " it doesn't exist and replace existing corrections for matching catalogs. See"
        " the help for --csv-output for more information on the correction values."
        " However, unlike the CSV output, the positional unit for the database is"
        " always degrees and the flux unit is always Jy[/beam]."
    ),
)
@click.option(
    "--plot-path",
    type=click.Path(),
    help=(
        "Save plots of the crossmatched sources positional offsets and flux ratios as a"
        " PNG image to the given directory. If the directory does not exist, it will be"
        " created. The axis units are specified by --positional-unit and --flux-unit."
        " The output filenames will be the name of the input catalog with the suffix"
        " _positional_offset.png and _flux_ratio.png."
    ),
)
@click.option(
    "--plot-pos-gridsize",
    type=click.INT,
    default=50,
    help=(
        "Number of hexagons in the x-direction of the positional offset plot. Default "
        "is 50."
    ),
)
def vast_xmatch_qc(
    reference_catalog_path: str,
    catalog_path: str,
    radius: Angle = Angle("10arcsec"),
    condon: bool = False,
    askap_database: Optional[Path] = None,
    psf_reference: Optional[Tuple[float, float]] = None,
    psf: Optional[Tuple[float, float]] = None,
    fix_m: bool = False,
    fix_b: bool = False,
    verbose: bool = False,
    aegean: bool = False,
    positional_unit: u.Unit = u.Unit("arcsec"),
    flux_unit: u.Unit = u.Unit("mJy"),
    csv_output: Optional[str] = None,
    sqlite_output: Optional[str] = None,
    plot_path: Optional[str] = None,
    plot_pos_gridsize: int = 50,
):
    if verbose:
        logger.setLevel(logging.DEBUG)
    logger.debug("Set logger to DEBUG.")

    # convert catalog path strings to Path objects
    reference_catalog_path = Path(reference_catalog_path)
    catalog_path = Path(catalog_path)
    # add beam divisor as we currently only work with peak fluxes
    flux_unit /= u.beam
    # get the PSF
    if askap_database is not None:
        obs_df = pd.DataFrame()
        for field_data_path in Path(askap_database).glob("epoch_*/field_data.csv"):
            df = pd.read_csv(field_data_path)
            obs_df = obs_df.append(df)
        racs_df = pd.read_csv(Path(askap_database).parent.parent / "racs/db/epoch_0/field_data.csv")
        racs_df["FIELD_NAME"] = racs_df.FIELD_NAME.str.replace("RACS_", "VAST_")
        obs_df = obs_df.append(racs_df)
        obs_df = obs_df.set_index(["FIELD_NAME", "SBID"]).sort_index()

        if psf is None:
            _, _, field, sbid, *_ = catalog_path.name.split(".")
            sbid = int(sbid[2:])
            obs_data = obs_df.loc[(field, sbid)]
            psf = (obs_data.PSF_MAJOR, obs_data.PSF_MINOR)
        if psf_reference is None:
            _, _, field_ref, sbid_ref, *_ = reference_catalog_path.name.split(".")
            sbid_ref = int(sbid_ref[2:])
            obs_data = obs_df.loc[(field_ref, sbid_ref)]
            psf_reference = (obs_data.PSF_MAJOR, obs_data.PSF_MINOR)

    reference_catalog = Catalog(
        reference_catalog_path,
        psf=psf_reference,
        condon=condon,
        input_format="aegean" if aegean else "selavy",
    )
    catalog = Catalog(
        catalog_path,
        psf=psf,
        condon=condon,
        input_format="aegean" if aegean else "selavy",
    )

    # CSV/SQLite output requires the catalogs to follow VAST naming conventions
    if (csv_output is not None or sqlite_output is not None) and (
        catalog.field is None or catalog.epoch is None
    ):
        e = UnknownFilenameConvention(
            f"Unknown catalog filename convention: {catalog.path}. "
            "Correction output is unavailable."
        )
        logger.error(e)
        raise SystemExit(e)

    # perform the crossmatch
    xmatch_qt = crossmatch_qtables(catalog, reference_catalog, radius=radius)

    # apply crossmatch filters
    mask_flux_err = xmatch_qt["flux_peak_err"] > 0
    mask_flux_err &= xmatch_qt["flux_peak_err_reference"] > 0
    logger.info(
        "Removing %d sources with peak flux errors = 0.",
        len(xmatch_qt) - sum(mask_flux_err),
    )

    mask_siblings = xmatch_qt["has_siblings"] == 0
    mask_siblings &= xmatch_qt["has_siblings_reference"] == 0
    logger.info(
        "Removing %d sources with siblings.", len(xmatch_qt) - sum(mask_siblings)
    )

    mask_nn = xmatch_qt["nn_separation"] >= (2.5 * u.arcmin)
    logger.info(
        "Removing %d sources with nearest neighbours within 2.5 arcmin.",
        len(xmatch_qt) - sum(mask_nn),
    )

    mask_snr = (xmatch_qt["flux_peak"] / xmatch_qt["rms_image"]).decompose() >= 10
    logger.info("Removing %d sources with SNR < 10.", len(xmatch_qt) - sum(mask_snr))

    mask = mask_flux_err & mask_siblings & mask_nn & mask_snr
    data = xmatch_qt[mask]
    logger.info(
        "%d crossmatched sources remaining (%d%%).",
        len(data),
        (len(data) / len(xmatch_qt)) * 100,
    )

    # calculate positional offsets and flux ratio
    dra_median, ddec_median, dra_madfm, ddec_madfm = calculate_positional_offsets(data)
    dra_median_value = dra_median.to(positional_unit).value
    dra_madfm_value = dra_madfm.to(positional_unit).value
    ddec_median_value = ddec_median.to(positional_unit).value
    ddec_madfm_value = ddec_madfm.to(positional_unit).value
    logger.info(
        "dRA median: %.2f MADFM: %.2f %s. dDec median: %.2f MADFM: %.2f %s.",
        dra_median_value,
        dra_madfm_value,
        positional_unit,
        ddec_median_value,
        ddec_madfm_value,
        positional_unit,
    )

    gradient, offset, gradient_err, offset_err = calculate_flux_offsets(
        data, fix_m=fix_m, fix_b=fix_b
    )
    ugradient = ufloat(gradient, gradient_err)
    uoffset = ufloat(offset.to(flux_unit).value, offset_err.to(flux_unit).value)
    logger.info(
        "ODR fit parameters: Sp = Sp,ref * %s + %s %s.", ugradient, uoffset, flux_unit
    )

    if csv_output is not None or sqlite_output is not None:
        # output has been requested
        flux_corr_mult = 1 / ugradient
        flux_corr_add = -1 * uoffset
        if csv_output is not None:
            csv_output_path = Path(csv_output)  # ensure Path object
            sbid = catalog.sbid if catalog.sbid is not None else ""
            if not csv_output_path.exists():
                f = open(csv_output_path, "w")
                print(
                    "field,release_epoch,sbid,ra_correction,dec_correction,ra_madfm,"
                    "dec_madfm,flux_peak_correction_multiplicative,flux_peak_correction_additive,"
                    "flux_peak_correction_multiplicative_err,flux_peak_correction_additive_err,"
                    "n_sources",
                    file=f,
                )
            else:
                f = open(csv_output_path, "a")
            logger.info(
                "Writing corrections CSV. To correct positions, add the corrections to"
                " the original source positions i.e. RA' = RA + ra_correction /"
                " cos(Dec). To correct fluxes, add the additive correction and multiply"
                " the result by the multiplicative correction i.e. S' ="
                " flux_peak_correction_multiplicative(S +"
                " flux_peak_correction_additive)."
            )
            print(
                f"{catalog.field},{catalog.epoch},{sbid},{dra_median_value * -1},"
                f"{ddec_median_value * -1},{dra_madfm_value},{ddec_madfm_value},"
                f"{flux_corr_mult.nominal_value},{flux_corr_add.nominal_value},"
                f"{flux_corr_mult.std_dev},{flux_corr_add.std_dev},{len(data)}",
                file=f,
            )
            f.close()
        if sqlite_output is not None:
            db.init_database(sqlite_output)
            with db.database:
                logger.info("Writing corrections to database %s.", sqlite_output)
                q = db.VastCorrection.replace(
                    vast_type=catalog.type,
                    field=catalog.field,
                    release_epoch=catalog.epoch,
                    sbid=catalog.sbid,
                    ra_correction=dra_median.to("deg").value * -1,
                    dec_correction=ddec_median.to("deg").value * -1,
                    ra_madfm=dra_madfm.to("deg").value,
                    dec_madfm=ddec_madfm.to("deg").value,
                    flux_peak_correction_multiplicative=flux_corr_mult.nominal_value,
                    flux_peak_correction_additive=(
                        flux_corr_add.nominal_value * flux_unit
                    )
                    .to("Jy/beam")
                    .value,
                    flux_peak_correction_multiplicative_err=flux_corr_mult.std_dev,
                    flux_peak_correction_additive_err=(
                        flux_corr_add.std_dev * flux_unit
                    )
                    .to("Jy/beam")
                    .value,
                    n_sources=len(data),
                )
                q.execute()

    if plot_path is not None:
        plot_path_path = Path(plot_path)  # ensure Path object
        plot_path_path.mkdir(parents=True, exist_ok=True)
        title: Optional[str] = None
        if (
            catalog.field
            and catalog.epoch
            and reference_catalog.field
            and reference_catalog.epoch
        ):
            title = (
                f"{catalog.field}.{catalog.epoch} X {reference_catalog.field}."
                f"{reference_catalog.epoch} (ref)"
            )
        g_pos_offset = positional_offset_plot(
            data,
            title=title,
            unit=positional_unit,
            offsets=(dra_median, ddec_median, dra_madfm, ddec_madfm),
            hex_gridsize=plot_pos_gridsize,
        )
        g_pos_offset.savefig(
            plot_path_path / f"{catalog.path.stem}_positional_offset.png",
            bbox_inches="tight",
        )

        ax_flux_ratio = flux_ratio_plot(
            data,
            title=title,
            unit=flux_unit,
            fit_params=(gradient, offset, gradient_err, offset_err),
        )
        ax_flux_ratio.figure.savefig(
            plot_path_path / f"{catalog.path.stem}_flux_ratio.png",
            bbox_inches="tight",
        )


@cli.command(
    name="export",
    help="Export the contents of a SQLite database of VAST corrections to a CSV file.",
    short_help="Export database corrections to CSV.",
)
@click.argument(
    "database-path",
    type=click.Path(exists=True, dir_okay=False),
)
@click.argument(
    "csv-path",
    type=click.Path(dir_okay=False, writable=True),
)
@click.option("--vast-type", type=click.Choice(Catalog.CATALOG_TYPES), default=None)
@click.option(
    "--positional-unit",
    type=ANGLE_UNIT_TYPE,
    default="arcsec",
    help=(
        "Positional correction output unit. Must be an angular unit. Default is arcsec."
    ),
)
@click.option(
    "--flux-unit",
    type=FLUX_UNIT_TYPE,
    default="mJy",
    help=(
        "Flux correction output unit. Must be a spectral flux density unit. Do not"
        " include a beam divisor, this will be automatically added for peak flux"
        " values. Default is mJy."
    ),
)
@click.option("-v", "--verbose", is_flag=True)
def vast_xmatch_export(
    database_path: str,
    csv_path: str,
    vast_type: Optional[str],
    positional_unit: u.Unit,
    flux_unit: u.Unit,
    verbose: bool = False,
):
    if verbose:
        logger.setLevel(logging.DEBUG)
    db.init_database(database_path)
    db.export_csv(csv_path, vast_type, positional_unit, flux_unit)
