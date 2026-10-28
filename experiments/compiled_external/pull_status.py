import os,subprocess
print("DONE" if os.path.exists("/content/run3_done") else "RUNNING", open("/content/run3_done").read() if os.path.exists("/content/run3_done") else "")
print(subprocess.run("pgrep -af run_remote | head -2; tail -n 6 /content/run3.log | cut -c1-250",shell=True,capture_output=True,text=True).stdout)
print("@@STATUS@@"); print(open("/content/status.json").read() if os.path.exists("/content/status.json") else "{}")
