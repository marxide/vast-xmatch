import logging
from pathlib import Path
from typing import Optional, Tuple, Union

from astropy.coordinates import Angle
from astropy.table import QTable
import astropy.units as u
import click
import matplotlib.pyplot as plt
import pandas as pd
from uncertainties import ufloat

from vast_xmatch.catalogs import (
    read_selavy,
    calculate_condon_flux_errors,
    get_psf_size_from_metadata_server,
    get_vast_filename_parts,
    MetadataNotFound,
    UnknownFilenameConvention,
)
from vast_xmatch.crossmatch import (
    crossmatch_qtables,
    calculate_positional_offsets,
    calculate_flux_offsets,
)
from vast_xmatch.plot import positional_offset_plot, flux_ratio_plot


class AngleType(click.ParamType):
    name = "angle"

    def convert(self, value, param, ctx):
        try:
            angle = Angle(value)
            return angle
        except ValueError:
            self.fail(f"astropy.coordinates.Angle does not understand: {value}.")


ANGLE_TYPE = AngleType()
logger = logging.getLogger("vast_xmatch")


def _transform_epoch_raw(epoch: str) -> str:
    epoch = epoch.replace("EPOCH", "")
    if epoch.startswith("0"):
        epoch = epoch.lstrip("0")
    return epoch


def _transform_epoch_vastp(epoch: str) -> str:
    return f"vastp{_transform_epoch_raw(epoch)}"


def common_options(function):
    function = click.argument(
        "reference_catalog", type=click.Path(exists=True, dir_okay=False)
    )(function)
    function = click.argument("catalog", type=click.Path(exists=True, dir_okay=False))(
        function
    )
    function = click.option(
        "--radius",
        type=ANGLE_TYPE,
        default="10 arcsec",
        help=(
            "Maximum separation limit for nearest-neighbour crossmatch. Accepts any "
            "string understood by astropy.coordinates.Angle."
        ),
    )(function)
    function = click.option(
        "--condon",
        is_flag=True,
        help="Calculate Condon (1997) flux errors and use them instead of the original errors.",
    )(function)
    function = click.option(
        "--lookup-psf",
        is_flag=True,
        help=(
            "If using --condon, lookup the catalog PSFs from the VAST metadata server. "
            "Requires the catalog files to be named according to the VAST convention, e.g. "
            "VAST_0102-06A.EPOCH01.I.selavy.components.txt."
        ),
    )(function)
    function = click.option(
        "--psf-reference",
        nargs=2,
        type=float,
        required=False,
        help=(
            "If using --condon and not using --lookup-psf, use this specified PSF size in "
            "arcsec for `reference_catalog`."
        ),
    )(function)
    function = click.option(
        "--psf",
        nargs=2,
        type=float,
        required=False,
        help=(
            "If using --condon and not using --lookup-psf, use this specified PSF size in "
            "arcsec for `catalog`."
        ),
    )(function)
    function = click.option("-v", "--verbose", is_flag=True)(function)
    return function


def vast_xmatch_selavy(
    reference_catalog: Union[Path, str],
    catalog: Union[Path, str],
    radius: Angle = Angle("10arcsec"),
    condon: bool = False,
    lookup_psf: bool = True,
    psf_reference: Optional[Tuple[float, float]] = None,
    psf: Optional[Tuple[float, float]] = None,
    verbose: bool = False,
) -> QTable:
    if verbose:
        logger.setLevel(logging.DEBUG)
    logger.debug("Set logger to DEBUG.")
    logger.debug("radius: %s.", radius)
    if isinstance(reference_catalog, str):
        reference_catalog = Path(reference_catalog)
    if isinstance(catalog, str):
        catalog = Path(catalog)

    # validate Condon and PSF options
    if condon:
        if not lookup_psf:
            try:
                if not psf_reference:
                    raise SystemExit(
                        "--psf-reference must be supplied if --lookup-psf is not used."
                    )
                if not psf:
                    raise SystemExit(
                        "--psf must be supplied if --lookup-psf is not used."
                    )
            except SystemExit as e:
                logger.exception(e)
                raise
            psf_major_ref, psf_minor_ref = psf_reference * u.arcsec
            psf_major, psf_minor = psf * u.arcsec
        else:
            # determine field and epoch from filenames
            try:
                # reference catalog
                _, parts_ref = get_vast_filename_parts(reference_catalog)
                psf_major_ref, psf_minor_ref = (
                    get_psf_size_from_metadata_server(
                        parts_ref["field"], _transform_epoch_raw(parts_ref["epoch"])
                    )
                    * u.arcsec
                )
                # catalog
                _, parts = get_vast_filename_parts(catalog)
                psf_major, psf_minor = (
                    get_psf_size_from_metadata_server(
                        parts["field"], _transform_epoch_raw(parts["epoch"])
                    )
                    * u.arcsec
                )
            except UnknownFilenameConvention as e:
                logger.exception(e)
                logger.error(
                    "Condon errors can only be calculated without --lookup-psf for "
                    "catalogs that follow the VAST filename conventions."
                )
                raise SystemExit("A fatal error occurred. Check the logs for details.")
            except MetadataNotFound as e:
                logger.exception(e)
                raise SystemExit("A fatal error occurred. Check the logs for details.")
        logger.info(
            "Reference catalog PSF size: %.2f, %.2f arcsec.",
            psf_major_ref.to("arcsec").value,
            psf_minor_ref.to("arcsec").value,
        )
        logger.info(
            "Catalog PSF size: %.2f, %.2f arcsec.",
            psf_major.to("arcsec").value,
            psf_minor.to("arcsec").value,
        )

    reference_catalog_qt = read_selavy(reference_catalog)
    catalog_qt = read_selavy(catalog)

    # compute Condon (1997) flux errors
    if condon:
        for qt, (_psf_major, _psf_minor) in zip(
            (reference_catalog_qt, catalog_qt),
            ((psf_major_ref, psf_minor_ref), (psf_major, psf_minor)),
        ):
            qt["flux_peak_err_condon"] = calculate_condon_flux_errors(
                qt,
                _psf_major,
                _psf_minor,
                correct_peak_for_noise=True,
            )
            qt["flux_peak_err_selavy"] = qt["flux_peak_err"]
            qt["flux_peak_err"] = qt["flux_peak_err_condon"]
            qt["flux_peak_selavy"] = qt["flux_peak"]
            qt["flux_peak"] = qt["flux_peak_condon"]

    # perform the crossmatch
    xmatch_qt = crossmatch_qtables(catalog_qt, reference_catalog_qt, radius=radius)
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

    return data


def vast_xmatch_qc(
    positional_unit: str = "arcsec",
    flux_unit: str = "mJy/beam",
    csv_output: Optional[str] = None,
    **kwargs,
):
    print("TEST")
    data = vast_xmatch_selavy(**kwargs)

    # calculate positional offsets and flux ratio
    df = pd.DataFrame(
        {
            "dra": data["dra"].to(positional_unit),
            "ddec": data["ddec"].to(positional_unit),
            "flux_peak": data["flux_peak"].to(flux_unit),
            "flux_peak_err": data["flux_peak_err"].to(flux_unit),
            "flux_peak_reference": data["flux_peak_reference"].to(flux_unit),
            "flux_peak_err_reference": data["flux_peak_err_reference"].to(flux_unit),
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
        _, parts = get_vast_filename_parts(kwargs["catalog"])
        sbid = parts["sbid"] if "sbid" in parts else ""
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
                "the original source positions i.e. RA' = RA + ra_correction. To correct "
                "fluxes, add the additive correction and multiply the result by the "
                "multiplicative correction i.e. S' = flux_peak_correction_multiplicative"
                "(S + flux_peak_correction_additive)."
            )
        flux_corr_mult = 1 / ugradient
        flux_corr_add = -1 * uoffset
        print(
            (
                f"{parts['field']},{parts['epoch']},{sbid},{dra_median * -1},"
                f"{ddec_median * -1},{dra_madfm},{ddec_madfm},{flux_corr_mult.nominal_value},"
                f"{flux_corr_add.nominal_value},{flux_corr_mult.std_dev},{flux_corr_add.std_dev}"
            ),
            file=f,
        )
        f.close()


@click.command()
@common_options
def vast_xmatch_selavy_plots(*args, **kwargs):
    data = vast_xmatch_selavy(*args, **kwargs)
    _ = positional_offset_plot(data, unit="arcsec")
    _ = flux_ratio_plot(data)
    plt.show()
