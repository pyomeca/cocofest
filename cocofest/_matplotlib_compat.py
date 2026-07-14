"""
Temporary fix for issue https://github.com/pyomeca/bioptim/pull/1059,
where get_cmap is no longer imported from matplotlib.cm but from matplotlib.plt
This fix should be removed once bioptim address it and release a new version
"""

import matplotlib.cm


def _get_cmap_compat(name=None, lut=None):
    import matplotlib

    cmap = matplotlib.colormaps[name] if name is not None else matplotlib.colormaps["viridis"]
    return cmap.resampled(lut) if lut is not None else cmap


if not hasattr(matplotlib.cm, "get_cmap"):
    matplotlib.cm.get_cmap = _get_cmap_compat
