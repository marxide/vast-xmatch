import logging
from pathlib import Path
from typing import Optional, Tuple, Union

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
def vast_xmatch_qc(
    reference_catalog_path: Union[Path, str],
    catalog_path: Union[Path, str],
    radius: Angle = Angle("10arcsec"),
    condon: bool = False,
    psf_reference: Optional[Tuple[float, float]] = None,
    psf: Optional[Tuple[float, float]] = None,
    verbose: bool = False,
    aegean: bool = False,
    positional_unit: str = "arcsec",
    flux_unit: str = "mJy",
    csv_output: Optional[str] = None,
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

    reference_catalog = Catalog(
        reference_catalog_path, psf=psf_reference, condon=condon
    )
    catalog = Catalog(
        catalog_path,
        psf=psf,
        condon=condon,
        input_format="aegean" if aegean else "selavy",
    )

    # CSV output requires the catalogs to follow VAST naming conventions
    if csv_output is not None and (catalog.field is None or catalog.epoch is None):
        e = UnknownFilenameConvention(
            f"Unknown catalog filename convention: {catalog.path}. CSV output is unavailable."
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
    df = pd.DataFrame(
        {
            "dra": data["dra"].to(positional_unit),
            "ddec": data["ddec"].to(positional_unit),
            "flux_peak": data["flux_peak"].to(flux_unit / u.beam),
            "flux_peak_err": data["flux_peak_err"].to(flux_unit / u.beam),
            "flux_peak_reference": data["flux_peak_reference"].to(flux_unit / u.beam),
            "flux_peak_err_reference": data["flux_peak_err_reference"].to(
                flux_unit / u.beam
            ),
            "flux_peak_ratio": data["flux_peak_ratio"],
        }
    )
    dra_median, ddec_median, dra_madfm, ddec_madfm = calculate_positional_offsets(df)
    logger.info(
        "dRA median: %.2f MADFM: %.2f %s. dDec median: %.2f MADFM: %.2f %s.",
        dra_median,
        dra_madfm,
        positional_unit,
        ddec_median,
        ddec_madfm,
        positional_unit,
    )

    gradient, offset, gradient_err, offset_err = calculate_flux_offsets(df)
    ugradient = ufloat(gradient, gradient_err)
    uoffset = ufloat(offset, offset_err)
    logger.info(
        "ODR fit parameters: Sp = Sp,ref * %s + %s %s.", ugradient, uoffset, flux_unit
    )

    if csv_output is not None:
        csv_output_path = Path(csv_output)
        sbid = catalog.sbid if catalog.sbid is not None else ""
        if not csv_output_path.exists():
            f = open(csv_output_path, "w")
            print(
                (
                    "field,release_epoch,sbid,ra_correction,dec_correction,ra_madfm,"
                    "dec_madfm,flux_peak_correction_multiplicative,flux_peak_correction_additive,"
                    "flux_peak_correction_multiplicative_err,flux_peak_correction_additive_err"
                ),
                file=f,
            )
        else:
            f = open(csv_output_path, "a")
        logger.info(
            "Writing corrections CSV. To correct positions, add the corrections to "
            "the original source positions i.e. RA' = RA + ra_correction / cos(Dec). To "
            "correct fluxes, add the additive correction and multiply the result by the "
            "multiplicative correction i.e. S' = flux_peak_correction_multiplicative"
            "(S + flux_peak_correction_additive)."
        )
        flux_corr_mult = 1 / ugradient
        flux_corr_add = -1 * uoffset
        print(
            (
                f"{catalog.field},{catalog.epoch},{sbid},{dra_median * -1},"
                f"{ddec_median * -1},{dra_madfm},{ddec_madfm},{flux_corr_mult.nominal_value},"
                f"{flux_corr_add.nominal_value},{flux_corr_mult.std_dev},{flux_corr_add.std_dev}"
            ),
            file=f,
        )
        f.close()
