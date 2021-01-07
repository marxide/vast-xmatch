import logging

__version__ = "0.0.1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-25s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
