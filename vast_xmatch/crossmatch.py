import logging
from typing import Tuple

from astropy.coordinates import SkyCoord, Angle, match_coordinates_sky
from astropy.table import QTable, join, join_skycoord
import astropy.units as u
import numpy as np
from scipy import odr

from vast_xmatch.catalogs import Catalog


logger = logging.getLogger(__name__)


def median_abs_deviation(data):
    median = np.median(data)
    return np.median(np.abs(data - median))


def straight_line(B, x):
    m, b = B
    return m * x + b


def join_match_coordinates_sky(
    coords1: SkyCoord, coords2: SkyCoord, seplimit: u.arcsec
):
    idx, separation, dist_3d = match_coordinates_sky(coords1, coords2)
    mask = separation < seplimit
    return np.where(mask)[0], idx[mask], separation[mask], dist_3d[mask]


def crossmatch_qtables(
    catalog: Catalog,
    catalog_reference: Catalog,
    radius: Angle = Angle("10 arcsec"),
    catalog_coord_cols: Tuple[str, str] = ("ra_deg_cont", "dec_deg_cont"),
    catalog_reference_coord_cols: Tuple[str, str] = ("ra_deg_cont", "dec_deg_cont"),
) -> QTable:
    catalog_ra, catalog_dec = catalog_coord_cols
    catalog_reference_ra, catalog_reference_dec = catalog_reference_coord_cols

    logger.debug("Using crossmatch radius: %s.", radius)

    xmatch = join(
        catalog.table,
        catalog_reference.table,
        keys="coord",
        table_names=["", "reference"],
        join_funcs={
            "coord": join_skycoord(radius, distance_func=join_match_coordinates_sky)
        },
    )
    # remove trailing _ from catalog column names
    xmatch.rename_columns(
        [col for col in xmatch.colnames if col.endswith("_")],
        [col.rstrip("_") for col in xmatch.colnames if col.endswith("_")],
    )
    # compute the separations
    xmatch["separation"] = xmatch["coord_reference"].separation(xmatch["coord"])
    xmatch["dra"], xmatch["ddec"] = xmatch["coord_reference"].spherical_offsets_to(
        xmatch["coord"]
    )
    xmatch["flux_peak_ratio"] = (
        xmatch["flux_peak"] / xmatch["flux_peak_reference"]
    ).decompose()

    logger.info(
        "Num cross-matches: %d. Num cross-matches to unique reference source: %d"
        " (%d%%).",
        len(xmatch),
        len(set(xmatch["coord_id"])),
        (len(set(xmatch["coord_id"])) / len(xmatch)) * 100,
    )

    return xmatch


def calculate_positional_offsets(
    xmatch_qt: QTable,
) -> Tuple[u.Quantity, u.Quantity, u.Quantity, u.Quantity]:
    """Calculate the median positional offsets and the median absolute deviation between
    matched sources.

    Parameters
    ----------
    xmatch_qt : QTable
        QTable of crossmatched sources. Must contain columns: dra, ddec.

    Returns
    -------
    Tuple[u.Quantity, u.Quantity, u.Quantity, u.Quantity]
        Median RA offset, median Dec offset, median absolute deviation of RA offsets,
        median absolute deviation of Dec offsets. Units match their inputs and are of
        angular type.
    """
    dra_median = np.median(xmatch_qt["dra"])
    dra_madfm = median_abs_deviation(xmatch_qt["dra"])
    ddec_median = np.median(xmatch_qt["ddec"])
    ddec_madfm = median_abs_deviation(xmatch_qt["ddec"])

    return dra_median, ddec_median, dra_madfm, ddec_madfm


def calculate_flux_offsets(
    xmatch_qt: QTable,
    init_m: float = 1.0,
    init_b: float = 0.0,
    fix_m: bool = False,
    fix_b: bool = False,
) -> Tuple[float, u.Quantity, float, u.Quantity]:
    """Calculate the gradient and offset of a straight-line fit to the peak fluxes for
    crossmatched sources. The function `y = mx + b` is fit to the reference peak fluxes
    vs the peak fluxes using orthogonal distance regression with `scipy.odr`.

    Parameters
    ----------
    xmatch_qt : QTable
        QTable of crossmatched sources. Must contain columns: flux_peak,
        flux_peak_reference, flux_peak_err, flux_peak_err_reference.
    init_m : float
        Initial gradient parameter passed to the fitting function, default 1.0.
    init_b : float
        Initial offset parameter passed to the fitting function, default 0.0.
    fix_m : bool
        If True, do not allow the gradient to vary during fitting, default False.
    fix_b : bool
        If True, do not allow the offest to vary during fitting, default False.

    Returns
    -------
    Tuple[float, u.Quantity, float, u.Quantity]
        Model fit parameters: the gradient, intercept (offset), gradient error, and
        intercept error. Offset and offset error unit match the reference flux peak
        input and are of spectral flux density type.
    """
    ifixb = [0 if fix_m else 1, 0 if fix_b else 1]
    flux_unit = xmatch_qt["flux_peak_reference"].unit
    linear_model = odr.Model(straight_line)
    # convert all to reference flux unit as ODR does not preserve Quantity objects
    odr_data = odr.RealData(
        xmatch_qt["flux_peak_reference"].to(flux_unit).value,
        xmatch_qt["flux_peak"].to(flux_unit).value,
        sx=xmatch_qt["flux_peak_err_reference"].to(flux_unit).value,
        sy=xmatch_qt["flux_peak_err"].to(flux_unit).value,
    )
    odr_obj = odr.ODR(odr_data, linear_model, beta0=[init_m, init_b], ifixb=ifixb)
    odr_out = odr_obj.run()
    gradient, offset = odr_out.beta
    gradient_err, offset_err = odr_out.sd_beta

    return gradient, offset * flux_unit, gradient_err, offset_err * flux_unit
