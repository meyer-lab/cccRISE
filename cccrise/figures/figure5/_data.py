"""Figure 5's shared names."""

LABELLINGS = ("coarse", "fine")
#: Shorthand for a method's components: C (cccRISE), P, T, S.
SHORT = {"cccRISE": "C", "PB": "P", "TC2C": "T", "STAC": "S"}


def short(method: str, column) -> str:
    return f"{SHORT[method]}{int(column)}"
