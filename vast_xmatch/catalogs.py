import logging
from pathlib import Path
import re
from typing import Tuple, Union, Dict
from urllib.parse import quote

from astropy.coordinates import SkyCoord
from astropy.table import QTable, join
import astropy.units as u
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SELAVY_COLUMN_UNITS = {
    "ra_deg_cont": u.deg,
    "dec_deg_cont": u.deg,
    "ra_err": u.arcsec,
    "dec_err": u.arcsec,
    "flux_peak": u.mJy / u.beam,
    "flux_peak_err": u.mJy / u.beam,
    "maj_axis": u.arcsec,
    "maj_axis_err": u.arcsec,
    "min_axis": u.arcsec,
    "min_axis_err": u.arcsec,
    "pos_ang": u.deg,
    "pos_ang_err": u.deg,
    "rms_image": u.mJy / u.beam,
}


class UnknownFilenameConvention(Exception):
    pass


class MetadataNotFound(Exception):
    pass


def _convert_selavy_columns_to_quantites(
    qt: QTable, units: Dict[str, u.Unit] = SELAVY_COLUMN_UNITS
) -> QTable:
    for col, unit in units.items():
        qt[col].unit = unit
    return qt


def get_psf_size_from_metadata_server(field: str, epoch: str) -> Tuple[float, float]:
    metadata = pd.read_json(
        (
            f"https://metadata.vast-survey.org/images?field_name={quote(field)}"
            f"&release_epoch={quote(epoch)}"
        )
    )
    if len(metadata) > 1:
        logger.warning(
            (
                "Metadata server unexpectedly returned more than one result for %s "
                "epoch %s. Using the first result.",
            ),
            field,
            epoch,
        )
    try:
        psf_major = metadata.loc[0, "psf_major"] * u.arcsec
        psf_minor = metadata.loc[0, "psf_minor"] * u.arcsec
    except KeyError:
        raise MetadataNotFound(
            f"Metadata server returned no results for {field} epoch {epoch}."
        )
    return psf_major, psf_minor


def get_vast_filename_parts(filename: Union[Path, str]) -> Tuple[str, Dict[str, str]]:
    if isinstance(filename, str):
        filename = Path(filename)
    pattern_combined = re.compile(
        r"^(?P<field>(?:RACS|VAST)_\d{4}[+-]\d{2}\w)\.(?P<epoch>EPOCH\d{2}x?)\."
        r"(?P<stokes>[IQUV])\.components$"
    )
    pattern_tile = re.compile(
        r"^selavy-image\.(?P<stokes>[iquv])\.SB(?P<sbid>\d+)\.cont\."
        r"(?P<field>(?:RACS|VAST)_\d{4}[+-]\d{2}\w)\.linmos\.taylor\.0\.restored\."
        r"components$"
    )

    match = pattern_combined.match(filename.stem)
    if match:
        logger.debug("Using COMBINED image filename convention regex pattern.")
        logger.debug("Parts: %s", match.groupdict())
        return "COMBINED", match.groupdict()

    match = pattern_tile.match(filename.stem)
    if match:
        logger.debug("Using TILE image filename convention regex pattern")
        tile_parts = match.groupdict()
        # try to get the epoch from the full path
        for path_part in filename.parts:
            if path_part.startswith("EPOCH"):
                tile_parts["epoch"] = path_part
                break
        else:
            # EPOCH wasn't in any of the path parts
            raise UnknownFilenameConvention(
                f"{filename.name} appears to be a TILE image but the epoch is not in "
                f"the provided path: {filename}."
            )
        logger.debug("Parts: %s", tile_parts)
        return "TILE", tile_parts

    raise UnknownFilenameConvention(
        f"Failed to identify the filename convention for {filename.name}."
    )


def read_selavy(catalog_path: Path) -> QTable:
    """Read a Selavy fixed-width component catalog and return a QTable.
    Assumed to contain at least the following columns with the given units:
        - `ra_deg_cont` and `dec_deg_cont`: degrees.
        - `ra_err` and `dec_err`: arcseconds.
        - `flux_peak` and `flux_peak_err`: mJy/beam.
        - `maj_axis`, `maj_axis_err`, `min_axis`, `min_axis_err`: arcseconds.
        - `pos_ang` and `pos_ang_err`: degrees.
        - `rms_image`: mJy/beam.
    These columns will be converted to Astropy quantites assuming the above units.

    Parameters
    ----------
    catalog_path : Path
        Path to the Selavy catalog file.

    Returns
    -------
    QTable
        Selavy catalog as a QTable, with extra columns:
        - `coord`: `SkyCoord` object of the source coordinate.
        - `nn_separation`: separation to the nearest-neighbour source as a Quantity with
            angular units.
    """
    df = pd.read_fwf(catalog_path, skiprows=[1]).drop(columns="#")
    qt = _convert_selavy_columns_to_quantites(QTable.from_pandas(df))
    qt["coord"] = SkyCoord(ra=qt["ra_deg_cont"], dec=qt["dec_deg_cont"])
    _, qt["nn_separation"], _ = qt["coord"].match_to_catalog_sky(
        qt["coord"], nthneighbor=2
    )
    return qt


def read_selavy_votable(catalog_path: Path) -> QTable:
    qt = QTable.read(catalog_path, format="votable")
    qt["coord"] = SkyCoord(ra=qt["ra_deg_cont"], dec=qt["dec_deg_cont"])
    _, qt["nn_separation"], _ = qt["coord"].match_to_catalog_sky(
        qt["coord"], nthneighbor=2
    )
    return qt


def read_hdf(catalog_path: Path) -> pd.DataFrame:
    df = pd.read_hdf(catalog_path, key="data")
    df["field"] = df.field.str.split(".", n=1, expand=True)[0]
    qt = _convert_selavy_columns_to_quantites(QTable.from_pandas(df))
    qt["coord"] = SkyCoord(ra=qt["ra_deg_cont"], dec=qt["dec_deg_cont"])
    _, qt["nn_separation"], _ = qt["coord"].match_to_catalog_sky(
        qt["coord"], nthneighbor=2
    )
    return qt


def read_aegean_csv(catalog_path: Path) -> QTable:
    """Read an Aegean CSV component catalog and return a QTable.
    Assumed to contain at least the following columns with the given units:
        - `ra` and `dec`: degrees.
        - `err_ra` and `err_dec`: degrees.
        - `peak_flux` and `err_peak_flux`: Jy/beam.
        - `a`, `err_a`, `b`, `err_b`: fitted semi-major and -minor axes in arcseconds.
        - `pa` and `err_pa`: degrees.
        - `local_rms`: Jy/beam.
    These columns will be converted to Astropy quantites assuming the above units.

    Parameters
    ----------
    catalog_path : Path
        Path to the Selavy catalog file.

    Returns
    -------
    QTable
        Aegean component catalog as a QTable, with extra columns:
        - `coord`: `SkyCoord` object of the source coordinate.
        - `nn_separation`: separation to the nearest-neighbour source as a Quantity with
            angular units.
    """
    AEGEAN_COLUMN_MAP = {
        # aegean name: (selavy name, aegean unit)
        "ra": ("ra_deg_cont", u.deg),
        "dec": ("dec_deg_cont", u.deg),
        "err_ra": ("ra_err", u.deg),
        "err_dec": ("dec_err", u.deg),
        "peak_flux": ("flux_peak", u.Jy / u.beam),
        "err_peak_flux": ("flux_peak_err", u.Jy / u.beam),
        "a": ("maj_axis", u.arcsec),
        "b": ("min_axis", u.arcsec),
        "pa": ("pos_ang", u.arcsec),
        "err_a": ("maj_axis_err", u.arcsec),
        "err_b": ("min_axis_err", u.deg),
        "err_pa": ("pos_ang_err", u.deg),
        "local_rms": ("rms_image", u.Jy / u.beam),
    }
    qt = QTable.read(catalog_path)
    # rename columns to match selavy convention and assign units
    for col, (new_col, unit) in AEGEAN_COLUMN_MAP.items():
        qt.rename_column(col, new_col)
        qt[new_col].unit = unit
    # add has_siblings column
    island_source_counts = qt[["island", "source"]].group_by("island").groups.aggregate(np.sum)
    island_source_counts.rename_column("source", "has_siblings")
    island_source_counts["has_siblings"] = island_source_counts["has_siblings"].astype(bool)
    qt = join(qt, island_source_counts, keys="island", join_type="left")

    qt["coord"] = SkyCoord(ra=qt["ra_deg_cont"], dec=qt["dec_deg_cont"])
    _, qt["nn_separation"], _ = qt["coord"].match_to_catalog_sky(
        qt["coord"], nthneighbor=2
    )
    return qt


def calculate_condon_flux_errors(
    catalog_qt: QTable,
    psf_major,
    psf_minor,
    alpha_maj1=2.5,
    alpha_min1=0.5,
    alpha_maj2=0.5,
    alpha_min2=2.5,
    alpha_maj3=1.5,
    alpha_min3=1.5,
    clean_bias=0.0,
    clean_bias_error=0.0,
    frac_flux_cal_error=0.0,
    correct_peak_for_noise=False,
):

    noise = catalog_qt["rms_image"]
    snr = catalog_qt["flux_peak"] / noise

    rho_sq3 = (
        (
            catalog_qt["maj_axis"]
            * catalog_qt["min_axis"]
            / (4.0 * psf_major * psf_minor)
        )
        * (1.0 + (psf_major / catalog_qt["maj_axis"]) ** 2) ** alpha_maj3
        * (1.0 + (psf_minor / catalog_qt["min_axis"]) ** 2) ** alpha_min3
        * snr ** 2
    )

    flux_peak_col = catalog_qt["flux_peak"]
    if correct_peak_for_noise:
        catalog_qt["flux_peak_condon"] = catalog_qt["flux_peak"] + (
            -(noise ** 2) / catalog_qt["flux_peak"] + clean_bias
        )
        flux_peak_col = catalog_qt["flux_peak_condon"]

    errorpeaksq = (
        (frac_flux_cal_error * flux_peak_col) ** 2
        + clean_bias_error ** 2
        + 2.0 * flux_peak_col ** 2 / rho_sq3
    )
    errorpeak = np.sqrt(errorpeaksq)

    return errorpeak
