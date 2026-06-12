"""
bioamla.system — system services (configuration, dependencies, environment).

Folds the former dependency / util services into plain raising functions and
dataclasses:

- :mod:`bioamla.system.dependency` — check/install system tools (FFmpeg,
  libsndfile, PortAudio).
- :mod:`bioamla.system.util` — version and compute-device information.

These power the ``bioamla config`` CLI group. ``torch`` is imported lazily by
:func:`bioamla.system.util.get_device_info`.
"""

from bioamla.system import dependency, util
from bioamla.system.dependency import DependencyInfo, DependencyReport
from bioamla.system.util import DeviceInfo, DevicesData, VersionData

__all__ = [
    "dependency",
    "util",
    "DependencyInfo",
    "DependencyReport",
    "DeviceInfo",
    "DevicesData",
    "VersionData",
]
