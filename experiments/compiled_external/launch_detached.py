"""Runs on the Colab VM (Amendment D): starts run_remote.py in a new session so a client disconnect does not stop it.
Output goes to /content/run3.log, the progress file is /content/status.json, /content/run3_done is written when the runner exits."""
import subprocess
import sys

cmd = "python3 /content/run_remote.py > /content/run3.log 2>&1; echo $? > /content/run3_done"
p = subprocess.Popen(["bash", "-c", cmd], start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("started pid", p.pid)
sys.stdout.flush()
