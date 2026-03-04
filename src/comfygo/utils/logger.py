import logging
import logging.handlers
import sys
import re
from pathlib import Path
from typing import Optional

from rich.logging import RichHandler
from .console import SmartHighlighter

TRACE = 5
logging.addLevelName(TRACE, "TRACE")

def _trace(self, msg, *args, **kwargs):
    if self.isEnabledFor(TRACE):
        self._log(TRACE, msg, args, **kwargs)

logging.Logger.trace = _trace

class DropTqdmNoise(logging.Filter):
    def __init__(self):
        super().__init__()
        # Adjust patterns to what you actually see
        self._pat = re.compile(r"(\b\d+%|\bETA\b|\bit/s\b|\b/s\b|█|▉|▊|▌)")

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        # Return False to drop the record from console
        return not bool(self._pat.search(msg))


class MaxLevelFilter(logging.Filter):
    def __init__(self, max_level: int):
        super().__init__()
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.max_level


def _rotating_file_handler(path: str, level: str | int) -> logging.Handler:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    h = logging.handlers.TimedRotatingFileHandler(
        filename=p,
        when="midnight",
        backupCount=7,
        encoding="utf-8",
        utc=True,
        delay=False,  # open file immediately
    )
    h.setLevel(level)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    return h


def configure_logging(
    *,
    level: str | int = "INFO",
    info_log: Optional[str] = None,
    info_level: str | int = "INFO",
    error_log: Optional[str] = None,
    error_level: str | int = "ERROR",
) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    rich_h = RichHandler(
        level=level,
        show_time=True,
        show_level=False,
        show_path=False,
        rich_tracebacks=True,
        highlighter=SmartHighlighter(),
        markup=False,
    )
    rich_h.addFilter(DropTqdmNoise())
    root.addHandler(rich_h)
    root.addHandler(rich_h)

    log = logging.getLogger(__name__)

    if info_log:
        try:
            info_h = _rotating_file_handler(info_log, info_level)
            info_h.addFilter(MaxLevelFilter(logging.WARNING))
            root.addHandler(info_h)

            # Force a write so you immediately know it worked
            log.info("info log initialized: %s", info_log)

        except Exception:
            log.exception("FAILED to initialize info_log: %s", info_log)

    if error_log:
        try:
            err_h = _rotating_file_handler(error_log, error_level)
            root.addHandler(err_h)

            # Force a write so you immediately know it worked
            log.error("error log initialized: %s", error_log)

        except Exception:
            log.exception("FAILED to initialize error_log: %s", error_log)

    _install_fatal_excepthook()


def _install_fatal_excepthook() -> None:
    log = logging.getLogger("fatal")

    def excepthook(exc_type, exc, tb):
        if exc_type is KeyboardInterrupt:
            sys.__excepthook__(exc_type, exc, tb)
            return
        log.critical("Unhandled fatal exception", exc_info=(exc_type, exc, tb))

    sys.excepthook = excepthook
