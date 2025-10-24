import re
import tempfile
from pathlib import Path
from time import sleep

from clearml import Task

task = Task.init()
logger = task.get_logger()

#----------------------------------------------------------------------------------------------------
# args for the running task
#----------------------------------------------------------------------------------------------------
args = {
    "target_project": "",
    "new_task_name": "",
    "queue_name": "",
}

args = task.connect(args)

#----------------------------------------------------------------------------------------------------
# Load bash script from file
#----------------------------------------------------------------------------------------------------
script = Path("tensorflow_clearml_script.sh").read_text()

#----------------------------------------------------------------------------------------------------
# Create a ClearML task with the modified bash script
#----------------------------------------------------------------------------------------------------
with tempfile.NamedTemporaryFile(mode="w+t", delete=True) as named_temp_file:

    # Write back script into temporary file
    named_temp_file.write(script)
    named_temp_file.seek(0)

    # Create a new ClearML task to be managed by this application with the bash script we composed
    bash_task = Task.create(
        project_name=args["target_project"],
        task_name=args["new_task_name"],
        script=named_temp_file.name,
        binary="/bin/bash",
        docker="nvidia/cuda:12.6.3-cudnn-runtime-ubuntu20.04",
        docker_args="-e CLEARML_AGENT_SLURM_SKIP_SRUN=1",
    )

#----------------------------------------------------------------------------------------------------
# Update task hyperparameters according to application wizard (these will be used by the Slurm template)
#----------------------------------------------------------------------------------------------------
#bash_task.set_parameters(
#    {
#        "properties/num_tasks": args["num_tasks"],
#        "properties/num_gpus": args["num_gpus"],
#        "properties/memory_gb": args["memory_gb"],
#    }
#)

#----------------------------------------------------------------------------------------------------
# Enqueue the new task to the queue requested in the application wizard
# (we assume there is a slurm agent monitoring that queue)
#----------------------------------------------------------------------------------------------------
Task.enqueue(task=bash_task, queue_name=args["queue_name"])

#----------------------------------------------------------------------------------------------------
# Monitor the task we just enqueued
#----------------------------------------------------------------------------------------------------
status = bash_task.status
while True:
    print("Monitoring Bash task")

    bash_task.reload()

    if bash_task.status != status:
        print(f"Task changed status to {bash_task.status}")
        status = bash_task.status

    if status not in ("queued", "in_progress"):
        print(f"Exiting since task status is {status}")
        break

    sleep(60)
