import logging
from pathlib import Path
from typing import Optional, Union

import astropy.units as u
import peewee

from vast_xmatch.catalogs import Catalog


database = peewee.SqliteDatabase(None)
logger = logging.getLogger(__name__)


class VastCorrection(peewee.Model):
    # (database value, display value)
    VAST_TYPE_CHOICES = tuple([(t, t) for t in Catalog.CATALOG_TYPES])

    vast_type = peewee.CharField(max_length=15, choices=VAST_TYPE_CHOICES)
    field = peewee.CharField(
        max_length=15,
        help_text="Field name including survey prefix e.g. VAST_0012+00A.",
    )
    release_epoch = peewee.CharField(
        max_length=8,
        help_text="Release epoch string including EPOCH prefix e.g. EPOCH05x.",
    )
    sbid = peewee.IntegerField(
        null=True,
        help_text="ASKAP SBID without the SB prefix. Only required for TILE images.",
    )
    ra_correction = peewee.FloatField()
    dec_correction = peewee.FloatField()
    ra_madfm = peewee.FloatField()
    dec_madfm = peewee.FloatField()
    flux_peak_correction_multiplicative = peewee.FloatField()
    flux_peak_correction_additive = peewee.FloatField()
    flux_peak_correction_multiplicative_err = peewee.FloatField()
    flux_peak_correction_additive_err = peewee.FloatField()
    n_sources = peewee.IntegerField(
        help_text="Number of crossmatched sources used to calculate corrections."
    )

    class Meta:
        database = database
        indexes = ((("vast_type", "field", "release_epoch"), True),)


def init_database(path: Union[Path, str]):
    database.init(str(path))
    with database:
        database.create_tables([VastCorrection])


def export_csv(
    output_path: Union[Path, str],
    vast_type: Optional[str],
    positional_unit: u.Unit,
    flux_unit: u.Unit,
):
    output_path = Path(output_path)
    # warn user if kind isn't recognized
    if vast_type not in Catalog.CATALOG_TYPES and vast_type is not None:
        logger.warning(
            "vast_type is set to %s. Must one of %s, or None. Setting to None.",
            vast_type,
            Catalog.CATALOG_TYPES,
        )
        vast_type = None

    corrections = VastCorrection.select()
    if vast_type:
        corrections = corrections.where(VastCorrection.vast_type == vast_type)

    with output_path.open("w") as output_fd:
        if vast_type is None:
            print("vast_type,", file=output_fd, end="")
        print(
            "field,release_epoch,sbid,ra_correction,dec_correction,ra_madfm,"
            "dec_madfm,flux_peak_correction_multiplicative,flux_peak_correction_additive,"
            "flux_peak_correction_multiplicative_err,flux_peak_correction_additive_err,n_sources",
            file=output_fd,
        )
        for correction in corrections:
            flux_corr_a = (
                (correction.flux_peak_correction_additive * u.Jy).to(flux_unit).value
            )
            flux_corr_a_err = (
                (correction.flux_peak_correction_additive_err * u.Jy)
                .to(flux_unit)
                .value
            )
            sbid = "" if correction.sbid is None else correction.sbid
            if vast_type is None:
                print(f"{correction.vast_type},", file=output_fd, end="")
            print(
                (
                    f"{correction.field},{correction.release_epoch},{sbid},"
                    f"{(correction.ra_correction * u.deg).to(positional_unit).value},"
                    f"{(correction.dec_correction * u.deg).to(positional_unit).value},"
                    f"{(correction.ra_madfm * u.deg).to(positional_unit).value},"
                    f"{(correction.dec_madfm * u.deg).to(positional_unit).value},"
                    f"{correction.flux_peak_correction_multiplicative},{flux_corr_a},"
                    f"{correction.flux_peak_correction_multiplicative_err},{flux_corr_a_err},"
                    f"{correction.n_sources}"
                ),
                file=output_fd,
            )
