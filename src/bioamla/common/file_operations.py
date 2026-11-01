"""Operations for ``util``: thin adapters over :mod:`bioamla.common.files`.

Downloading fetches any URL, so it is local-only.
"""

from __future__ import annotations

from typing import Any

from bioamla.common.config import config_dataclass
from bioamla.operations import OperationContext, PathStr, operation


@config_dataclass
class DownloadRequest:
    """Download a URL into a folder.

    Attributes:
        url: The URL.
        output_dir: The folder.
    """

    url: str
    output_dir: PathStr = "."


@operation("util.download", request=DownloadRequest, kind="job", remote=False)
def download_op(request: DownloadRequest, context: OperationContext) -> dict[str, Any]:
    """Download a URL into a folder (local only)."""
    from bioamla.common.files import download_to_dir

    return {"url": request.url, "output": download_to_dir(request.url, request.output_dir)}


@config_dataclass
class UnzipRequest:
    """Extract a ZIP archive.

    Attributes:
        file: The archive.
        output_dir: Where to extract it.
    """

    file: PathStr
    output_dir: PathStr = "."


@operation("util.unzip", request=UnzipRequest, kind="job")
def unzip_op(request: UnzipRequest, context: OperationContext) -> dict[str, Any]:
    """Extract a ZIP archive."""
    from bioamla.common.files import unzip_to

    unzip_to(request.file, request.output_dir)
    return {"file": request.file, "output": request.output_dir}


@config_dataclass
class ZipRequest:
    """Zip a folder or a file.

    Attributes:
        source: The folder or file.
        output: The archive to write.
    """

    source: PathStr
    output: PathStr


@operation("util.zip", request=ZipRequest, kind="job")
def zip_op(request: ZipRequest, context: OperationContext) -> dict[str, Any]:
    """Zip a folder or a file."""
    from bioamla.common.files import zip_path

    zip_path(request.source, request.output)
    return {"source": request.source, "output": request.output}
