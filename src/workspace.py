"""Private, short-lived folders for the documents people upload.

Each browser session gets its own folder. It is deleted when the user replaces the documents, and
whenever any session starts or reruns, the folders of sessions that are no longer connected (a
closed tab, a reload, a new session) are deleted too. A server restart removes everything the
earlier run left behind. Nothing a user uploads is kept on the server after their session.

    ROOT / <server process id> / <session id> / <one folder per ingest>
"""

import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(tempfile.gettempdir()) / "ragforge_workspaces"


def connected_sessions() -> set[str] | None:
    """The ids of the Streamlit sessions that are connected right now, or None when Streamlit's
    session list cannot be read (then nothing is removed on the strength of it)."""
    try:
        from streamlit.runtime import Runtime

        manager = Runtime.instance()._session_mgr
        return {info.session.id for info in manager.list_active_sessions()}
    except Exception:  # noqa: BLE001 - a different Streamlit version, or no server running
        return None


def process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # it exists, and belongs to someone else
        return True
    return True


class Workspaces:
    def __init__(self, root: Path | None = None, pid: int | None = None):
        self.root = root or ROOT
        self.mine = self.root / str(pid or os.getpid())

    def folder_for(self, session_id: str) -> Path:
        """This session's private folder (created if needed)."""
        folder = self.mine / session_id
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def remove(self, session_id: str) -> None:
        shutil.rmtree(self.mine / session_id, ignore_errors=True)

    def remove_disconnected(self, connected: set[str] | None) -> list[str]:
        """Delete the folders of sessions that are not connected any more, and everything an
        earlier server process left behind. Returns the names it deleted."""
        removed = []
        if self.root.exists():
            for other in self.root.iterdir():  # a folder per server process
                if other.name != self.mine.name and not (
                    other.name.isdigit() and process_is_running(int(other.name))
                ):
                    shutil.rmtree(other, ignore_errors=True)
                    removed.append(other.name)
        if connected is not None and self.mine.exists():
            for folder in self.mine.iterdir():
                if folder.name not in connected:
                    shutil.rmtree(folder, ignore_errors=True)
                    removed.append(folder.name)
        return removed
