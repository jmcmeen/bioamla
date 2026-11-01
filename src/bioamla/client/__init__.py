"""
bioamla.client — work in a bioamla server's workspaces from another machine.

:class:`Client` talks to a server started with ``bioamla serve``: create and list
workspaces, upload and download files, run operations (waiting for jobs and following
their progress) and manage jobs. Named remotes (``[remotes.<name>]`` in the user's
settings file, with the token in an environment variable) are managed with
:func:`add_remote` and opened with :func:`connect`.

It needs only ``requests``, part of the base install, so a laptop without the
``server`` extra or any machine-learning extras can drive a server.

Example:
    >>> from bioamla.client import add_remote, connect
    >>> add_remote("lab", "http://lab-server:8765")  # token in $BIOAMLA_LAB_TOKEN
    >>> lab = connect("lab")
    >>> lab.upload("costa-rica", "rec.wav", "raw/rec.wav")
    >>> lab.run("costa-rica", "detect.energy", {"file": "raw/rec.wav"})["count"]
"""

from bioamla.client._client import (
    FINISHED,
    Client,
    JobFailed,
    RemoteInvalidInput,
    RemoteNotFound,
)
from bioamla.client._remotes import (
    Remote,
    add_remote,
    connect,
    get_remote,
    remotes,
    remove_remote,
)
from bioamla.exceptions import RemoteError

__all__ = [
    "FINISHED",
    "Client",
    "JobFailed",
    "Remote",
    "RemoteError",
    "RemoteInvalidInput",
    "RemoteNotFound",
    "add_remote",
    "connect",
    "get_remote",
    "remotes",
    "remove_remote",
]
