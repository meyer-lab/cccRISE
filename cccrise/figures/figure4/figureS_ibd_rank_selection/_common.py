"""The IBD RISE factor matrices' component axis: 60 components, labeled
every `TICK_STEP`."""

TICK_STEP = 10


def component_ticks(ax, n: int) -> None:
    ticks = [1, *range(TICK_STEP, n + 1, TICK_STEP)]
    ax.set_xticks([t - 1 for t in ticks], ticks)
