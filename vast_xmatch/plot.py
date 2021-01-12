import logging
from typing import Optional, Tuple

from astropy.coordinates import Angle
from astropy.table import QTable
import astropy.units as u
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
    offsets: Optional[Tuple[u.Quantity, u.Quantity, u.Quantity, u.Quantity]] = None,
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
    offsets : Optional[Tuple[u.Quantity, u.Quantity, u.Quantity, u.Quantity]], optional
        Tuple of Quantity objects of angular unit type (dRA median, dDec median,
        dRA MADFM, dDec MADFM). If `None`, these values will be calculated. Default None.

    Returns
    -------
    sns.JointGrid
    """
    if offsets is None:
        dra_median, ddec_median, dra_madfm, ddec_madfm = calculate_positional_offsets(
            xmatch_qt
        )
        logger.info(
            "dRA median: %.2f MADFM: %.2f %s. dDec median: %.2f MADFM: %.2f %s.",
            dra_median.to(unit).value,
            dra_madfm.to(unit).value,
            unit,
            ddec_median.to(unit).value,
            ddec_madfm.to(unit).value,
            unit,
        )
    else:
        dra_median, ddec_median, dra_madfm, ddec_madfm = offsets
    dra_median_value = dra_median.to(unit).value
    dra_madfm_value = dra_madfm.to(unit).value
    ddec_median_value = ddec_median.to(unit).value
    ddec_madfm_value = ddec_madfm.to(unit).value

    # seaborn expects a Pandas DataFrame
    df = pd.DataFrame(
        {
            "dra": xmatch_qt["dra"].to(unit).value,
            "ddec": xmatch_qt["ddec"].to(unit).value,
        }
    )
    data_min = min(df.dra.min(), df.ddec.min())
    data_max = max(df.dra.max(), df.ddec.max())
    g = sns.jointplot(
        x="dra",
        y="ddec",
        data=df,
        kind="hex",
        xlim=(data_min, data_max),
        ylim=(data_min, data_max),
        joint_kws=dict(
            gridsize=50,
            bins=10,
        ),
        marginal_kws=dict(edgecolor="none"),
    )
    g.set_axis_labels(f"∆RA ({unit})", f"∆Dec ({unit})")

    median_style = dict(color="black", linestyle="dashed", linewidth=1)
    madfm_style = dict(color="black", linestyle="dotted", linewidth=1)

    median_line_artist = g.ax_joint.axvline(dra_median_value, **median_style)
    _ = g.ax_joint.axhline(ddec_median_value, **median_style)
    _ = g.ax_marg_x.axvline(dra_median_value, **median_style)
    _ = g.ax_marg_y.axhline(ddec_median_value, **median_style)

    madfm_line_artist = g.ax_joint.axvline(dra_median_value - dra_madfm_value / 2, **madfm_style)
    _ = g.ax_joint.axvline(dra_median_value + dra_madfm_value / 2, **madfm_style)
    _ = g.ax_joint.axhline(ddec_median_value - ddec_madfm_value / 2, **madfm_style)
    _ = g.ax_joint.axhline(ddec_median_value + ddec_madfm_value / 2, **madfm_style)
    _ = g.ax_marg_x.axvline(dra_median_value - dra_madfm_value / 2, **madfm_style)
    _ = g.ax_marg_x.axvline(dra_median_value + dra_madfm_value / 2, **madfm_style)
    _ = g.ax_marg_y.axhline(ddec_median_value - ddec_madfm_value / 2, **madfm_style)
    _ = g.ax_marg_y.axhline(ddec_median_value + ddec_madfm_value / 2, **madfm_style)

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

    # place legend upper right corner at the top right of the figure
    # bbox_to_anchor necessary as the suptitle below may expand the figure bbox height to 1.05
    g.fig.legend(legend_handles, legend_labels, loc="upper right", bbox_to_anchor=(1.0, 1.0))

    if title:
        g.fig.suptitle(title, y=1.05)
    g.fig.tight_layout()

    return g


def flux_ratio_plot(
    xmatch_qt: QTable,
    title: Optional[str] = None,
    unit: str = "mJy/beam",
    fit_params: Optional[Tuple[float, u.Quantity, float, u.Quantity]] = None,
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
    fit_params : Optional[Tuple[float, u.Quantity, float, u.Quantity]], optional
        Tuple of (gradient, offset, gradient error, offset error). Gradient and gradient
        error are floats. Offset and offset error are Quantity objects with units of
        spectral flux density type that will be converted to `unit`. If `None`, these
        values will be calculated. Default None.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if fit_params is None:
        gradient, offset, gradient_err, offset_err = calculate_flux_offsets(xmatch_qt)
    else:
        gradient, offset, gradient_err, offset_err = fit_params
    ugradient = ufloat(gradient, gradient_err)
    uoffset = ufloat(offset.to(unit).value, offset_err.to(unit).value)
    if fit_params is None:
        logger.info(
            "ODR fit parameters: Sp = Sp,ref * %s + %s %s.", ugradient, uoffset, unit
        )

    # seaborn expects a Pandas DataFrame
    df = pd.DataFrame(
        {
            "flux_peak": xmatch_qt["flux_peak"].to(unit).value,
            "flux_peak_err": xmatch_qt["flux_peak_err"].to(unit).value,
            "flux_peak_reference": xmatch_qt["flux_peak_reference"].to(unit).value,
            "flux_peak_err_reference": xmatch_qt["flux_peak_err_reference"].to(unit).value,
            "flux_peak_ratio": xmatch_qt["flux_peak_ratio"],
        }
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
