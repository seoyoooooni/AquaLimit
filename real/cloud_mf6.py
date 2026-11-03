"""Use the official MODFLOW 6 Linux executable on hosted Streamlit runs."""

import gzip
import os
from pathlib import Path
import platform
import shutil
import tempfile


def configure_mf6():
    """Set MF6_EXE before real_model is imported; leave local installs alone."""
    if os.environ.get("MF6_EXE") or platform.system() != "Linux":
        return
    if platform.machine() not in {"x86_64", "AMD64"}:
        raise RuntimeError("Bundled MODFLOW 6 requires Linux x86_64")
    asset = Path(__file__).parent / "bin" / "mf6_6.7.0_linux_x86_64.gz"
    target = Path(tempfile.gettempdir()) / "aqualimit-mf6-6.7.0"
    if not target.is_file():
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix="aqualimit-mf6-", delete=False) as tmp:
            temporary = Path(tmp.name)
            try:
                with gzip.open(asset, "rb") as source:
                    shutil.copyfileobj(source, tmp)
                tmp.flush()
                os.fchmod(tmp.fileno(), 0o755)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
        os.replace(temporary, target)
    os.environ["MF6_EXE"] = str(target)
