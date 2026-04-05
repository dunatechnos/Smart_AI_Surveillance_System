import sys

# Emoji prefixes for log types
INFO_EMOJI = "ℹ️"
SUCCESS_EMOJI = "✅"
PROGRESS_EMOJI = "📡"
ERROR_EMOJI = "❌"

def info(message: str):
    print(f"{INFO_EMOJI} {message}")

def success(message: str):
    print(f"{SUCCESS_EMOJI} {message}")

def progress(source: str, current: int, total: int):
    percent = (current / total) * 100 if total else 0
    print(f"\r{PROGRESS_EMOJI} {source} | {current}/{total} | {percent:.1f}%", end="", flush=True)

def error(message: str):
    print(f"{ERROR_EMOJI} {message}", file=sys.stderr)
