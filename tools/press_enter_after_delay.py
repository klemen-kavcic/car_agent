"""Send one Enter keypress to the focused Windows application after a delay.

Usage:
    python tools/press_enter_after_delay.py 30

Focus the intended application during the countdown. Press Ctrl+C in this
terminal to cancel before the key is sent.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import time


VK_RETURN = 0x0D
KEYEVENTF_KEYUP = 0x0002


def press_enter() -> None:
    """Use the Windows keyboard API, without requiring pyautogui/pynput."""
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.keybd_event(VK_RETURN, 0, 0, 0)
    user32.keybd_event(VK_RETURN, 0, KEYEVENTF_KEYUP, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seconds", type=float, help="Delay before pressing Enter.")
    args = parser.parse_args()
    if args.seconds < 0:
        parser.error("seconds must be zero or positive")
    if os.name != "nt":
        parser.error("Run this with Windows Python, not WSL/Linux Python.")

    print(f"Enter will be sent in {args.seconds:g} seconds. Focus the target window now; Ctrl+C cancels.")
    time.sleep(args.seconds)
    press_enter()
    print("Enter sent.")


if __name__ == "__main__":
    main()
