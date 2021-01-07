import logging
from typing import Optional, Tuple

from astropy.coordinates import Angle
from astropy.table import QTable
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns
from uncertainties import ufloat, unumpy

from vast_xmatch.crossmatch import (
    calculate_positional_offsets,
    calculate_flux_offsets,
    straight_line,
)

logger = logging.getLogger(__name__)


def positional_offset_plot(
    xmatch_qt: QTable,
    title: Optional[str] = None,
    unit: str = "arcsec",
    pixel_size: Optional[Angle] = Angle("2.5arcsec"),
    offsets: Optional[Tuple[float, float, float, float]] = None,
) -> sns.JointGrid:
    """Plot the positional offsets of crossmatched sources.

    Parameters
    ----------
    xmatch_qt : QTable
        Crossmatched sources. Expected format is the output of
        `vast_xmatch.crossmatch.crossmatch_qtables`.
    title : Optional[str], optional
        Plot title, by default None
    unit : str, optional
        Plot units, by default "arcsec"
    pixel_size : Optional[Angle], optional
        Angular pixel size of the cataog source images. When supplied, the plot will
        contain a patch of this size centered at 0,0. Default Angle("2.5arcsec")
    offsets : Optional[Tuple[float, float, float, float]], optional
        Tuple of floats (dRA median, dDec median, dRA MADFM, dDec MADFM) in units of
        `unit`. If `None`, these values will be calculated. Default None

    Returns
    -------
    sns.JointGrid
    """
    # seaborn expects a Pandas DataFrame
    df = pd.DataFrame(
        {
            "dra": xmatch_qt["dra"].to(unit),
            "ddec": xmatch_qt["ddec"].to(unit),
        }
    )
    g = sns.jointplot(
        x="dra",
        y="ddec",
        data=df,
        kind="hex",
        joint_kws=dict(
            gridsize=50,
            bins=10,
        ),
        marginal_kws=dict(edgecolor="none"),
    )
    g.set_axis_labels(f"∆RA ({unit})", f"∆Dec ({unit})")

    if offsets is None:
        dra_median, ddec_median, dra_madfm, ddec_madfm = calculate_positional_offsets(
            df
        )
    else:
        dra_median, ddec_median, dra_madfm, ddec_madfm = offsets
    logger.info(
        "dRA median: %.2f MADFM: %.2f %s. dDec median: %.2f MADFM: %.2f %s.",
        dra_median,
        dra_madfm,
        unit,
        ddec_median,
        ddec_madfm,
        unit,
    )

    median_style = dict(color="black", linestyle="dashed", linewidth=1)
    madfm_style = dict(color="black", linestyle="dotted", linewidth=1)

    median_line_artist = g.ax_joint.axvline(dra_median, **median_style)
    _ = g.ax_joint.axhline(ddec_median, **median_style)
    _ = g.ax_marg_x.axvline(dra_median, **median_style)
    _ = g.ax_marg_y.axhline(ddec_median, **median_style)

    madfm_line_artist = g.ax_joint.axvline(dra_median - dra_madfm / 2, **madfm_style)
    _ = g.ax_joint.axvline(dra_median + dra_madfm / 2, **madfm_style)
    _ = g.ax_joint.axhline(ddec_median - ddec_madfm / 2, **madfm_style)
    _ = g.ax_joint.axhline(ddec_median + ddec_madfm / 2, **madfm_style)
    _ = g.ax_marg_x.axvline(dra_median - dra_madfm / 2, **madfm_style)
    _ = g.ax_marg_x.axvline(dra_median + dra_madfm / 2, **madfm_style)
    _ = g.ax_marg_y.axhline(ddec_median - ddec_madfm / 2, **madfm_style)
    _ = g.ax_marg_y.axhline(ddec_median + ddec_madfm / 2, **madfm_style)

    legend_handles = [median_line_artist, madfm_line_artist]
    legend_labels = ["Median", "MADFM"]

    if pixel_size is not None:
        logger.debug("Plotting pixel size: %s.", pixel_size.to_string())
        pixel_size = pixel_size.to(unit).value
        pixel_patch = Rectangle(
            (-pixel_size / 2, -pixel_size / 2),
            pixel_size,
            pixel_size,
            edgecolor="none",
            facecolor="orange",
            alpha=0.2,
        )
        g.ax_joint.add_patch(pixel_patch)
        legend_handles.append(pixel_patch)
        legend_labels.append("Pixel size")

    if title:
        g.fig.suptitle(title, y=1.05)

    g.fig.legend(legend_handles, legend_labels)
    g.fig.tight_layout()

    return g


def flux_ratio_plot(
    xmatch_qt: QTable,
    title: Optional[str] = None,
    unit: str = "mJy/beam",
    fit_params: Optional[Tuple[float, float, float, float]] = None,
) -> matplotlib.axes.Axes:
    """Plot the flux ratios of crossmatched sources. X-axis is the reference catalog
    flux, Y-axis is the catalog flux / reference catalog flux.

    Parameters
    ----------
    xmatch_qt : QTable
        Crossmatched sources. Expected format is the output of
        `vast_xmatch.crossmatch.crossmatch_qtables`.
    title : Optional[str], optional
        Plot title, by default None
    unit : str, optional
        Plot units, by default "mJy/beam"
    fit_params : Optional[Tuple[float, float, float, float]], optional
        Tuple of floats (gradient, offset, gradient error, offset error) in units of
        `unit`. If `None`, these values will be calculated. Default None

    Returns
    -------
    matplotlib.axes.Axes
    """
    # seaborn expects a Pandas DataFrame
    df = pd.DataFrame(
        {
            "flux_peak": xmatch_qt["flux_peak"].to(unit),
            "flux_peak_err": xmatch_qt["flux_peak_err"].to(unit),
            "flux_peak_reference": xmatch_qt["flux_peak_reference"].to(unit),
            "flux_peak_err_reference": xmatch_qt["flux_peak_err_reference"].to(unit),
            "flux_peak_ratio": xmatch_qt["flux_peak_ratio"],
        }
    )
    if fit_params is None:
        gradient, offset, gradient_err, offset_err = calculate_flux_offsets(df)
    else:
        gradient, offset, gradient_err, offset_err = fit_params
    ugradient = ufloat(gradient, gradient_err)
    uoffset = ufloat(offset, offset_err)
    logger.info(
        "ODR fit parameters: Sp = Sp,ref * %s + %s %s.", ugradient, uoffset, unit
    )

    fig, (ax, ax_resid) = plt.subplots(
        nrows=2, sharex=True, gridspec_kw=dict(height_ratios=[3, 1])
    )
    ax = sns.regplot(
        x="flux_peak_reference",
        y="flux_peak_ratio",
        data=df,
        marker=".",
        ci=None,
        fit_reg=False,
        scatter_kws=dict(
            s=5,
            color="0.7",
        ),
        ax=ax,
    )
    x = np.geomspace(df.flux_peak_reference.min(), df.flux_peak_reference.max())
    y = straight_line((ugradient, uoffset), x) / x
    ax.plot(
        x,
        unumpy.nominal_values(y),
        "-",
        color="C0",
        linewidth=2,
        label=f"ODR Fit: Sp = Sp,ref * {ugradient} + {uoffset}",
    )
    ax.fill_between(
        x,
        unumpy.nominal_values(y) - (unumpy.std_devs(y) / 2),
        unumpy.nominal_values(y) + (unumpy.std_devs(y) / 2),
    )
    ax.axhline(y=1.0, color="black", linestyle="dotted", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"$S_{\mathrm{p,reference}}$" + f"({unit})")
    ax.set_ylabel(
        r"$S_{\mathrm{p}}$ ratio ($S_{\mathrm{p}} / S_{\mathrm{p,reference}}$)"
    )
    ax.legend()
    if title:
        ax.set_title(title)

    # residual plot
    ax_resid.scatter(
        df.flux_peak_reference,
        ((df.flux_peak - offset) / gradient) / df.flux_peak_reference,
        s=5,
        marker=".",
        color="0.7",
    )
    ax_resid.axhline(y=1.0, color="black", linestyle="dotted", linewidth=1)
    return ax
