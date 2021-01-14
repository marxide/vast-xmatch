import peewee

from vast_xmatch.catalogs import Catalog


database = peewee.SqliteDatabase(None)


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
        indexes = (
            (("vast_type", "field", "release_epoch"), True),
        )


def init_database(name: str):
    database.init(name)
    with database:
        database.create_tables([VastCorrection])
