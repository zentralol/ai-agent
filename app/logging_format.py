"""ANSI-colored console logging for zentra_agent."""

from __future__ import annotations

import logging
import os
import re
import sys
from typing import Final

# ANSI escape sequences (no external deps).
RESET: Final = "\033[0m"
BOLD: Final = "\033[1m"
DIM: Final = "\033[2m"
ITALIC: Final = "\033[3m"

FG_CYAN: Final = "\033[36m"
FG_GREEN: Final = "\033[32m"
FG_YELLOW: Final = "\033[33m"
FG_RED: Final = "\033[31m"
FG_MAGENTA: Final = "\033[35m"
FG_BLUE: Final = "\033[34m"
FG_GRAY: Final = "\033[90m"

_LEVEL_STYLES: Final[dict[str, str]] = {
    "DEBUG": FG_GRAY,
    "INFO": FG_GREEN,
    "WARNING": f"{BOLD}{FG_YELLOW}",
    "ERROR": f"{BOLD}{FG_RED}",
    "CRITICAL": f"{BOLD}{FG_RED}",
}

# key=value tokens, including single/double-quoted values.
_KV_TOKEN_RE = re.compile(
    r"(\w+)=("
    r"'[^']*'"
    r'|"[^"]*"'
    r"|[^\s]+"
    r")"
)

# Standalone event names (first token before key=value pairs).
_EVENT_NAME_RE = re.compile(r"^([\w]+)(?=\s|$)")

_ERROR_EVENT_SUFFIXES = ("_unconfigured", "_bad_status", "_failed")
_DIM_BLUE_KEYS = frozenset({"request_id", "conversation_id", "user"})
_DIM_KEYS = frozenset({"level"})

_FORCE_COLOR_VARS = ("FORCE_COLOR", "LOG_COLOR")


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no"}


def use_color() -> bool:
    """Return True when ANSI colors should be emitted."""

    if os.environ.get("NO_COLOR"):
        return False
    if any(_env_truthy(name) for name in _FORCE_COLOR_VARS):
        return True
    return sys.stdout.isatty()


def _wrap(style: str, text: str, *, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{style}{text}{RESET}"


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _style_event_name(name: str, *, enabled: bool) -> str:
    if name == "tool_call_start":
        return _wrap(f"{BOLD}{FG_CYAN}", name, enabled=enabled)
    if name == "tool_call_end":
        return _wrap(f"{BOLD}{FG_YELLOW}", name, enabled=enabled)
    if "_tool_" in name:
        return _wrap(FG_CYAN, name, enabled=enabled)
    if any(name.endswith(suffix) for suffix in _ERROR_EVENT_SUFFIXES):
        return _wrap(f"{BOLD}{FG_RED}", name, enabled=enabled)
    return _wrap(f"{BOLD}{FG_CYAN}", name, enabled=enabled)


def _style_kv_pair(key: str, value: str, *, enabled: bool) -> str:
    if not enabled:
        return f"{key}={value}"

    if key == "event":
        event_name = _strip_quotes(value)
        styled_value = _wrap(
            f"{BOLD}{FG_CYAN}",
            value if value.startswith(("'", '"')) else event_name,
            enabled=True,
        )
        styled_key = _wrap(DIM, key, enabled=True)
        return f"{styled_key}={styled_value}"

    if key == "tool":
        styled_value = _wrap(f"{BOLD}{FG_MAGENTA}", value, enabled=True)
        return f"{key}={styled_value}"

    if key == "status":
        status_styles = {
            "success": f"{BOLD}{FG_GREEN}",
            "warning": f"{BOLD}{FG_YELLOW}",
            "error": f"{BOLD}{FG_RED}",
        }
        bare = _strip_quotes(value)
        style = status_styles.get(bare, "")
        styled_value = _wrap(style, value, enabled=bool(style)) if style else value
        return f"{key}={styled_value}"

    if key == "duration_ms":
        styled_value = _wrap(f"{DIM}{ITALIC}", value, enabled=True)
        return f"{key}={styled_value}"

    if key in _DIM_BLUE_KEYS:
        styled_key = _wrap(f"{DIM}{FG_BLUE}", key, enabled=True)
        styled_value = _wrap(f"{DIM}{FG_BLUE}", value, enabled=True)
        return f"{styled_key}={styled_value}"

    if key in _DIM_KEYS:
        styled_key = _wrap(DIM, key, enabled=True)
        styled_value = _wrap(DIM, value, enabled=True)
        return f"{styled_key}={styled_value}"

    is_quoted = (
        (value.startswith("'") and value.endswith("'"))
        or (value.startswith('"') and value.endswith('"'))
    )
    if is_quoted:
        styled_value = _wrap(ITALIC, value, enabled=True)
        styled_key = _wrap(DIM, key, enabled=True)
        return f"{styled_key}={styled_value}"

    styled_key = _wrap(DIM, key, enabled=True)
    return f"{styled_key}={value}"


def colorize_message(message: str, *, enabled: bool | None = None) -> str:
    """Apply ANSI styling to key=value log messages."""

    if not message:
        return message

    color_enabled = use_color() if enabled is None else enabled
    if not color_enabled:
        return message

    event_match = _EVENT_NAME_RE.match(message)
    if event_match:
        event_name = event_match.group(1)
        rest = message[event_match.end() :]
        if not rest or rest[0].isspace():
            styled_event = _style_event_name(event_name, enabled=True)
            return styled_event + _colorize_kv_tokens(rest, enabled=True)

    return _colorize_kv_tokens(message, enabled=True)


def _colorize_kv_tokens(text: str, *, enabled: bool) -> str:
    if not enabled or "=" not in text:
        return text

    parts: list[str] = []
    last_end = 0
    for match in _KV_TOKEN_RE.finditer(text):
        parts.append(text[last_end : match.start()])
        key, value = match.group(1), match.group(2)
        parts.append(_style_kv_pair(key, value, enabled=enabled))
        last_end = match.end()
    parts.append(text[last_end:])
    return "".join(parts)


class ColoredLogFormatter(logging.Formatter):
    """Formatter that adds ANSI colors to log record fields."""

    def format(self, record: logging.LogRecord) -> str:
        color_enabled = use_color()
        asctime = self.formatTime(record, self.datefmt)
        levelname = record.levelname
        logger_name = record.name
        message = record.getMessage()

        if not color_enabled:
            return f"{asctime} {levelname} {logger_name} {message}"

        level_style = _LEVEL_STYLES.get(levelname, "")
        colored_time = _wrap(DIM, asctime, enabled=True)
        colored_level = _wrap(level_style, levelname, enabled=bool(level_style))
        colored_name = _wrap(f"{DIM}{FG_GRAY}", logger_name, enabled=True)
        colored_message = colorize_message(message, enabled=True)
        return f"{colored_time} {colored_level} {colored_name} {colored_message}"


def build_colored_formatter() -> ColoredLogFormatter:
    """Return the shared console formatter for zentra_agent logs."""

    return ColoredLogFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def configure_structlog(*, formatter: ColoredLogFormatter | None = None) -> None:
    """Bridge structlog loggers through stdlib logging with the same colors."""

    import structlog

    shared_formatter = formatter or build_colored_formatter()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.KeyValueRenderer(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # structlog loggers (e.g. app.agent.runner) propagate to root.
    root = logging.getLogger()
    if not root.handlers:
        root.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(shared_formatter)
        root.addHandler(handler)


def suppress_noisy_loggers() -> None:
    """Turn down third-party loggers that flood the console at INFO."""

    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
