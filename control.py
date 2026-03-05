#!/usr/bin/env python3
"""Animal Farm Service Control Panel — run with: sudo python3 control.py"""

import curses
import subprocess
import sys

# NOTE: your list included "ollama-cpp-server" but the actual systemd service
# is "llama-cpp-server" — using the real name here.
SERVICES = [
    "face-api",
    "pose-api",
    "sam3-api",
    "claude-api",
    "clip-score",
    "BLIP-api",
    "CLIP-api",
    "colors-api",
    "detectron-api",
    "llama-api",
    "llama-cpp-api",
    "llama-cpp-server",
    "metadata-api",
    "moondream-api",
    "nsfw-api",
    "nudenet-api",
    "ocr-api",
    "ollama",
    "rtdetr-api",
    "rtdetr2-api",
    "rtmdet-api",
    "segmentation",
    "speciesnet-api",
    "windmill-api",
    "yolo-365-api",
    "yolo-api",
    "yolo-oi7-api",
]


def get_status(service):
    result = subprocess.run(
        ["systemctl", "is-active", service],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def run_action(service, action):
    result = subprocess.run(
        ["systemctl", action, service],
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stderr.strip()


def main(stdscr):
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN, -1)              # active
    curses.init_pair(2, curses.COLOR_RED, -1)                # failed
    curses.init_pair(3, curses.COLOR_YELLOW, -1)             # transitioning
    curses.init_pair(4, curses.COLOR_WHITE, -1)              # inactive / unknown
    curses.init_pair(5, curses.COLOR_BLACK, curses.COLOR_WHITE)  # selected row

    services = sorted(SERVICES, key=str.casefold)
    selected = 0
    statuses = {svc: "..." for svc in services}
    message = ""

    def refresh_statuses():
        for svc in services:
            statuses[svc] = get_status(svc)

    def draw():
        stdscr.clear()
        height, width = stdscr.getmaxyx()

        title = " Animal Farm Service Control "
        stdscr.addstr(0, max(0, (width - len(title)) // 2), title, curses.A_BOLD)
        stdscr.addstr(1, 0, "─" * (width - 1))

        for i, svc in enumerate(services):
            y = i + 2
            if y >= height - 3:
                break

            status = statuses.get(svc, "unknown")

            if status == "active":
                indicator, color = "●", curses.color_pair(1)
            elif status == "failed":
                indicator, color = "!", curses.color_pair(2)
            elif status in ("activating", "deactivating", "reloading"):
                indicator, color = "◌", curses.color_pair(3)
            elif status == "...":
                indicator, color = "·", curses.color_pair(4)
            else:
                indicator, color = "○", curses.color_pair(4)

            if i == selected:
                row = f" {indicator} {svc:<22} {status:<16}"
                stdscr.addstr(y, 0, row[: width - 1].ljust(width - 1), curses.color_pair(5))
            else:
                stdscr.addstr(y, 0, " ")
                stdscr.addstr(y, 1, indicator, color)
                stdscr.addstr(y, 3, f"{svc:<22} {status:<16}"[: width - 4])

        stdscr.addstr(height - 2, 0, "─" * (width - 1))
        if message:
            stdscr.addstr(height - 2, 2, f" {message} "[: width - 4], curses.A_BOLD)
        footer = "[s]tart  [S]top  [r]estart  [u]pdate  [q]uit"
        stdscr.addstr(height - 1, 0, footer[: width - 1])

        stdscr.refresh()

    refresh_statuses()

    while True:
        draw()
        key = stdscr.getch()

        if key == curses.KEY_UP:
            selected = max(0, selected - 1)
            message = ""
        elif key == curses.KEY_DOWN:
            selected = min(len(services) - 1, selected + 1)
            message = ""
        elif key == ord("q"):
            break
        elif key == ord("u"):
            message = "Refreshing..."
            draw()
            refresh_statuses()
            message = "Statuses refreshed"
        elif key == ord("s"):
            svc = services[selected]
            message = f"Starting {svc}..."
            draw()
            rc, err = run_action(svc, "start")
            statuses[svc] = get_status(svc)
            message = f"Started {svc}" if rc == 0 else f"Error: {err[:50]}"
        elif key == ord("S"):
            svc = services[selected]
            message = f"Stopping {svc}..."
            draw()
            rc, err = run_action(svc, "stop")
            statuses[svc] = get_status(svc)
            message = f"Stopped {svc}" if rc == 0 else f"Error: {err[:50]}"
        elif key == ord("r"):
            svc = services[selected]
            message = f"Restarting {svc}..."
            draw()
            rc, err = run_action(svc, "restart")
            statuses[svc] = get_status(svc)
            message = f"Restarted {svc}" if rc == 0 else f"Error: {err[:50]}"


if __name__ == "__main__":
    curses.wrapper(main)
