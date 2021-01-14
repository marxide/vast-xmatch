"""Traverse the VAST pilot data release directory and crossmatch each TILE and COMBINED
catalog with the equivalent catalog from EPOCH08 (the reference epoch). Write the
positional and flux corrections to a SQLite database.
"""
from itertools import chain
import multiprocessing as mp
from pathlib import Path
from typing import Optional

from vast_xmatch.cli import vast_xmatch_qc
from vast_xmatch.catalogs import get_vast_filename_parts


VAST_DATA_ROOT = Path("/raid-17/LS/kaplan/VAST")
DATABASE = "vast_corrections.db"


def qc(args):
    reference_catalog, catalog = args
    vast_xmatch_qc(
        [
            str(reference_catalog),
            str(catalog),
            "--condon",
            f"--sqlite-output={DATABASE}",
        ],
        standalone_mode=False,
    )


def find_reference_for_catalog(catalog: Path) -> Optional[Path]:
    vast_type, parts = get_vast_filename_parts(catalog)
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
    combined_catalogs = VAST_DATA_ROOT.glob("EPOCH*/COMBINED/STOKESI_SELAVY/*.components.txt")
    catalogs = chain(tile_catalogs, combined_catalogs)

    for catalog in catalogs:
        reference_catalog = find_reference_for_catalog(catalog)
        if reference_catalog:
            catalog_pairs.append((str(reference_catalog), str(catalog)))

    print(f"Number of QC runs: {len(catalog_pairs)}")
    pool = mp.Pool(processes=mp.cpu_count())
    pool.map(qc, catalog_pairs)
    pool.close()
