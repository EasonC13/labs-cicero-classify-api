import os
import subprocess
import signal
import time
from core.config import (
    API_PORT,
    API_HOST,
    API_WORKER,
    SLEEP_INTERVAL_SECOND,
    ANACONDA_ENV_NAME,
    PATH,
    MONGODB_PORT,
    MONGODB_PATH,
    HOST_A_MONGODB,
    MONGODB_HOST,
)

slient_commands = [
    f"mongod --port {MONGODB_PORT} --bind_ip {MONGODB_HOST} --dbpath {MONGODB_PATH}" if HOST_A_MONGODB else "",
]
    
commands = [
    f"""cd {PATH};
    eval "$(conda shell.bash hook)";
    conda activate {ANACONDA_ENV_NAME};
    python trainer/NER_trainer_runner.py;""",

    f"""cd {PATH};
    eval "$(conda shell.bash hook)";
    conda activate {ANACONDA_ENV_NAME};
    uvicorn app:app --port {API_PORT} --host {API_HOST} --workers {API_WORKER};""",
]

processes = []

for command in slient_commands:
    if command != "":
        print(command, end = "\n---\n")
        process = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        processes.append(process)


for command in commands:
    if command != "":
        print(command, end = "\n---\n")
        process = subprocess.Popen(command, shell=True)
        processes.append(process)


while True:
    try:
        time.sleep(SLEEP_INTERVAL_SECOND)
    except KeyboardInterrupt:
        for process in processes:
            process.send_signal(signal.SIGINT)
        for process in processes:
            process.wait()
        break
