#!/usr/bin/env python3
"""
PTC Launcher — Starts the Streamlit chemistry app and opens the browser.

This is the PyInstaller entry point. It launches Streamlit as a subprocess
pointing to the bundled app.py, then opens the default browser.
"""
import os
import sys
import subprocess
import webbrowser
import time
import socket


def find_free_port():
    """Find a free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def main():
    # Determine paths
    if getattr(sys, "frozen", False):
        # Running as PyInstaller bundle
        base_dir = sys._MEIPASS
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    app_path = os.path.join(base_dir, "ptc_app", "app.py")
    port = find_free_port()

    print(f"Starting PTC — Persistence Theory Chemistry")
    print(f"  App: {app_path}")
    print(f"  Port: {port}")
    print(f"  Opening browser at http://localhost:{port}")

    # Launch Streamlit
    env = os.environ.copy()
    env["PYTHONPATH"] = base_dir + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run",
            app_path,
            "--server.port", str(port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
            "--theme.primaryColor", "#1f77b4",
        ],
        env=env,
        cwd=base_dir,
    )

    # Wait a moment then open browser
    time.sleep(3)
    webbrowser.open(f"http://localhost:{port}")

    print(f"\nPTC is running. Close this window to stop the server.")
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        print("\nPTC stopped.")


if __name__ == "__main__":
    main()
