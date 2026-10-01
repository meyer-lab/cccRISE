"""Component match ambiguity: matched against runner-up FMS, from
``uv run cccrise-stability``."""

from .. import common, style

DATASET = "alad"


def plot(fig):
    ax = fig.subplots()
    common.plot_match_ambiguity(common.load_stability(DATASET), ax)
    ax.set_title("Match ambiguity")


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["half"])
