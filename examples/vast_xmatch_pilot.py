"""Traverse the VAST pilot data release directory and crossmatch each TILE and COMBINED
catalog with the equivalent catalog from EPOCH08 (the reference epoch). Write the
positional and flux corrections to a SQLite database.
"""
from itertools import chain
import multiprocessing as mp
from pathlib import Path
from typing import Optional

import pandas as pd
from vast_xmatch.cli import vast_xmatch_qc, transform_epoch_raw
from vast_xmatch.catalogs import Catalog, get_vast_filename_parts


VAST_DATA_ROOT = Path("/raid-17/LS/kaplan/VAST")
DATABASE = "vast_corrections.db"


def qc(args):
    (
        reference_catalog,
        reference_psf_major,
        reference_psf_minor,
        catalog,
        psf_major,
        psf_minor,
    ) = args
    arg_list = [
        str(reference_catalog),
        str(catalog),
        "--condon",
        "--psf-reference",
        f"{reference_psf_major}",
        f"{reference_psf_minor}",
        "--psf",
        f"{psf_major}",
        f"{psf_minor}",
        f"--sqlite-output={DATABASE}",
    ]
    vast_xmatch_qc(arg_list, standalone_mode=False)


def find_reference_for_catalog(catalog: Path) -> Optional[Path]:
    vast_type, parts = get_vast_filename_parts(catalog)
    if vast_type == Catalog.CATALOG_TYPE_TILE:
        vast_type += "S"
    reference_catalog: Optional[Path]
    try:
        reference_catalog = list(
            (VAST_DATA_ROOT / Path(f"EPOCH08/{vast_type}/STOKESI_SELAVY")).glob(
                f"*{parts['field'].replace('RACS', 'VAST')}.*.components.txt"
            )
        )[0]
    except IndexError:
        print(
            f"Could not find reference catalog for field {parts['field']}. Path: {catalog}"
        )
        reference_catalog = None
    return reference_catalog


if __name__ == "__main__":
    catalog_pairs = []
    tile_catalogs = VAST_DATA_ROOT.glob("EPOCH*/TILES/STOKESI_SELAVY/*.components.txt")
    combined_catalogs = VAST_DATA_ROOT.glob(
        "EPOCH*/COMBINED/STOKESI_SELAVY/*.components.txt"
    )
    catalogs = chain(tile_catalogs, combined_catalogs)

    psf_df = pd.read_json("https://metadata.vast-survey.org/images")[
        ["field_name", "release_epoch", "psf_major", "psf_minor", "psf_pa"]
    ].set_index(["field_name", "release_epoch"])

    for catalog in catalogs:
        reference_catalog = find_reference_for_catalog(catalog)
        if reference_catalog:
            _, ref_parts = get_vast_filename_parts(reference_catalog)
            reference_psf_major, reference_psf_minor = psf_df.loc[
                (ref_parts["field"], transform_epoch_raw(ref_parts["epoch"])),
                ("psf_major", "psf_minor"),
            ]
            _, parts = get_vast_filename_parts(catalog)
            psf_major, psf_minor = psf_df.loc[
                (parts["field"], transform_epoch_raw(parts["epoch"])),
                ("psf_major", "psf_minor"),
            ]
            catalog_pairs.append(
                (
                    str(reference_catalog),
                    f"{reference_psf_major:.2f}",
                    f"{reference_psf_minor:.2f}",
                    str(catalog),
                    f"{psf_major:.2f}",
                    f"{psf_minor:.2f}",
                )
            )

    print(f"Number of QC runs: {len(catalog_pairs)}")
    pool = mp.Pool(processes=mp.cpu_count())
    pool.map(qc, catalog_pairs)
    pool.close()
