import logging


class BullseyeLogFilter(logging.Filter):
    """Ensure consistent 'bullseye.*' logger naming in emitted records (no-op mapper)."""

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        # Currently a pass-through; kept for future normalization needs
        return True


def install_bullseye_log_filter() -> None:
    filt = BullseyeLogFilter()
    # root and common uvicorn loggers
    for logger_name in ('', 'uvicorn', 'uvicorn.error', 'uvicorn.access'):
        lg = logging.getLogger(logger_name)
        try:
            for h in list(lg.handlers):
                try:
                    h.addFilter(filt)
                except Exception:
                    pass
            lg.addFilter(filt)
        except Exception:
            pass


def sanitize_bullseye_loggers() -> None:
    """Remove custom handlers on 'bullseye.*' loggers and enable propagation."""
    try:
        # Iterate through all known loggers
        for name in list(logging.root.manager.loggerDict.keys()):  # type: ignore[attr-defined]
            if isinstance(name, str) and name.startswith('bullseye'):
                lg = logging.getLogger(name)
                # Remove all attached handlers
                for h in list(lg.handlers):
                    try:
                        lg.removeHandler(h)
                    except Exception:
                        pass
                # Ensure records flow to root and don't reformat here
                lg.propagate = True
                lg.setLevel(logging.NOTSET)
    except Exception:
        pass
