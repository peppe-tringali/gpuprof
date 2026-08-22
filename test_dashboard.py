"""Minimum dashboard test.

    python test_dashboard.py

Opens an in-process dashboard on a free port (URL prints at startup),
runs a 30-second mocked loop so the charts have something to draw,
then holds the process open so you can keep exploring the browser.
Ctrl-C to exit.
"""
import os, time

os.environ.setdefault("GPUPROF_MOCK", "1")
os.environ.setdefault("GPUPROF_MOCK_GPUS", "4")

import gpuprof

with gpuprof.profile("dashboard-test", dashboard=True, auto=False):
    print("\n↑ open that URL in a browser now.\n"
          "  Running 30 steps, ~1s each — you'll see charts update live.\n")
    for i in range(30):
        time.sleep(1)
        print(f"  step {i+1}/30", end="\r", flush=True)

print("\nDone. Dashboard still reachable at the URL above — Ctrl-C to exit.")
try:
    while True: time.sleep(3600)
except KeyboardInterrupt:
    pass
