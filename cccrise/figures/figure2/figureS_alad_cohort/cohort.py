"""Placeholder for a schematic of the ALAD cohort."""

from ... import common

AXES = ("full",)


def plot(axs):
    (ax,) = axs
    common.placeholder(
        ax,
        "ALAD cohort: 45 lung transplant recipients, 47 BAL samples\n"
        "(17 control, 30 ALAD: 17 declined, 13 recovered)",
    )


def makeFigure():
    return common.standalone_fixed(__name__)
