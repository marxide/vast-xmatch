import logging
from pathlib import Path
from typing import Optional, Tuple, Union

from astropy.coordinates import Angle
import astropy.units as u
import click
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
                f"{unit} is a {unit.physical_type} unit. It must be of type {unit_physical_type}."
            )
        else:
            return unit


class AngleUnitType(_AstropyUnitType):
    name = "angle unit"

    def convert(self, value, param, ctx):
        return super().convert(value, param, ctx, "angle")


class FluxUnitType(_AstropyUnitType):
    name = "flux unit"

    def convert(self, value, param, ctx):
        return super().convert(value, param, ctx, "spectral flux density")


class AngleQuantityType(click.ParamType):
    name = "angle quantity"

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


def _transform_epoch_raw(epoch: str) -> str:
    epoch = epoch.replace("EPOCH", "")
    if epoch.startswith("0"):
        epoch = epoch.lstrip("0")
    return epoch


def _transform_epoch_vastp(epoch: str) -> str:
    return f"vastp{_transform_epoch_raw(epoch)}"


@click.command()
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
        "the PSF sizes must be supplied using --psf-reference and/or --psf. The "
        "deafult behaviour is to lookup the PSF sizes from the VAST metadata server."
    ),
)
@click.option(
    "--psf-reference",
    nargs=2,
    type=float,
    required=False,
    help=(
        "If using --condon and not using --lookup-psf, use this specified PSF size in "
        "arcsec for `reference_catalog`."
    ),
)
@click.option(
    "--psf",
    nargs=2,
    type=float,
    required=False,
    help=(
        "If using --condon and not using --lookup-psf, use this specified PSF size in "
        "arcsec for `catalog`."
    ),
)
@click.option("-v", "--verbose", is_flag=True)
@click.option("--aegean", is_flag=True, help="Input catalog is an Aegean CSV.")
@click.option(
    "--positional-unit",
    type=ANGLE_UNIT_TYPE,
    default="arcsec",
    help="Positional correction output unit. Must be an angular unit. Default is arcsec.",
)
@click.option(
    "--flux-unit",
    type=FLUX_UNIT_TYPE,
    default="mJy",
    help=(
        "Flux correction output unit. Must be a spectral flux density unit. Do not "
        "include a beam divisor, this will be automatically added for peak flux values. "
        "Default is mJy."
    ),
)
@click.option(
    "--csv-output",
    type=click.Path(dir_okay=False, writable=True),
    help=(
        "Path to write CSV of positional and flux corrections. Only available if "
        "`catalog` follows VAST naming conventions as the field and epoch must be known. "
        "If the file exists, the corrections will be appended. Corrections are written "
        "in the units specified by --positional-unit and --flux-unit. To apply the "
        "corrections, use the following equations: ra corrected = ra + ra_correction / "
        "cos(dec); dec corrected = dec + dec_correction; flux peak corrected = "
        "flux_peak_correction_multiplicative * (flux peak + flux_peak_correction_additive. "
        "Note that these correction values have been modified to suit these equations "
        "and are different from the fitted values shown in the logs."
    ),
)
@click.option(
    "--sqlite-output",
    type=click.Path(dir_okay=False, file_okay=True),
    help=(
        "Write corrections to the given SQLite3 database. Will create the database if "
        "it doesn't exist and replace existing corrections for matching catalogs. See "
        "the help for --csv-output for more information on the correction values. "
        "However, unlike the CSV output, the positional unit for the database is always "
        "degrees and the flux unit is always Jy[/beam]."
    ),
)
@click.option(
    "--plot-path",
    type=click.Path(),
    help=(
        "Save plots of the crossmatched sources positional offsets and flux ratios as a "
        "PNG image to the given directory. If the directory does not exist, it will be "
        "created. The axis units are specified by --positional-unit and --flux-unit. "
        "The output filenames will be the name of the input catalog with the suffix "
        "_positional_offset.png and _flux_ratio.png."
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
    reference_catalog_path: Union[Path, str],
    catalog_path: Union[Path, str],
    radius: Angle = Angle("10arcsec"),
    condon: bool = False,
    psf_reference: Optional[Tuple[float, float]] = None,
    psf: Optional[Tuple[float, float]] = None,
    verbose: bool = False,
    aegean: bool = False,
    positional_unit: Union[str, u.Unit] = "arcsec",
    flux_unit: Union[str, u.Unit] = "mJy",
    csv_output: Optional[Union[Path, str]] = None,
    sqlite_output: Optional[Union[Path, str]] = None,
    plot_path: Optional[Union[Path, str]] = None,
    plot_pos_gridsize: int = 50,
):
    if verbose:
        logger.setLevel(logging.DEBUG)
    logger.debug("Set logger to DEBUG.")

    if isinstance(reference_catalog_path, str):
        reference_catalog_path = Path(reference_catalog_path)
    if isinstance(catalog_path, str):
        catalog_path = Path(catalog_path)
    if isinstance(psf_reference, tuple) and len(psf_reference) == 0:
        psf_reference = None
    if isinstance(psf, tuple) and len(psf) == 0:
        psf = None
    if isinstance(positional_unit, str):
        positional_unit = u.Unit(positional_unit)
    if isinstance(flux_unit, str):
        flux_unit = u.Unit(flux_unit)
    flux_unit /= u.beam  # add beam divisor as we currently only work with peak fluxes

    reference_catalog = Catalog(
        reference_catalog_path, psf=psf_reference, condon=condon
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
    # select xmatches with non-zero flux errors and no siblings
    logger.info("Removing crossmatched sources with siblings or flux peak errors = 0.")
    mask = xmatch_qt["flux_peak_err"] > 0
    mask &= xmatch_qt["flux_peak_err_reference"] > 0
    mask &= xmatch_qt["has_siblings"] == 0
    mask &= xmatch_qt["has_siblings_reference"] == 0
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

    gradient, offset, gradient_err, offset_err = calculate_flux_offsets(data)
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
            csv_output = Path(csv_output)  # ensure Path object
            sbid = catalog.sbid if catalog.sbid is not None else ""
            if not csv_output.exists():
                f = open(csv_output, "w")
                print(
                    "field,release_epoch,sbid,ra_correction,dec_correction,ra_madfm,"
                    "dec_madfm,flux_peak_correction_multiplicative,flux_peak_correction_additive,"
                    "flux_peak_correction_multiplicative_err,flux_peak_correction_additive_err",
                    file=f,
                )
            else:
                f = open(csv_output, "a")
            logger.info(
                "Writing corrections CSV. To correct positions, add the corrections to "
                "the original source positions i.e. RA' = RA + ra_correction / cos(Dec). To "
                "correct fluxes, add the additive correction and multiply the result by the "
                "multiplicative correction i.e. S' = flux_peak_correction_multiplicative"
                "(S + flux_peak_correction_additive)."
            )
            print(
                (
                    f"{catalog.field},{catalog.epoch},{sbid},{dra_median_value * -1},"
                    f"{ddec_median_value * -1},{dra_madfm_value},{ddec_madfm_value},"
                    f"{flux_corr_mult.nominal_value},{flux_corr_add.nominal_value},"
                    f"{flux_corr_mult.std_dev},{flux_corr_add.std_dev}"
                ),
                file=f,
            )
            f.close()
        if sqlite_output is not None:
            db.init_database(str(sqlite_output))
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
                    ).to("Jy/beam").value,
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
        plot_path = Path(plot_path)  # ensure Path object
        title: Optional[str] = None
        if (
            catalog.field
            and catalog.epoch
            and reference_catalog.field
            and reference_catalog.epoch
        ):
            title = (
                f"{catalog.field}.{catalog.epoch} X {reference_catalog.field}."
                f"{reference_catalog.epoch}"
            )
        g_pos_offset = positional_offset_plot(
            data,
            title=title,
            unit=positional_unit,
            offsets=(dra_median, ddec_median, dra_madfm, ddec_madfm),
            hex_gridsize=plot_pos_gridsize,
        )
        g_pos_offset.savefig(
            plot_path / f"{catalog.path.stem}_positional_offset.png",
            bbox_inches="tight",
        )

        ax_flux_ratio = flux_ratio_plot(
            data,
            title=title,
            unit=flux_unit,
            fit_params=(gradient, offset, gradient_err, offset_err),
        )
        ax_flux_ratio.figure.savefig(
            plot_path / f"{catalog.path.stem}_flux_ratio.png",
            bbox_inches="tight",
        )
