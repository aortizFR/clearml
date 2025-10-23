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
    "num_mpi_threads": 8,
    "num_openmp_threads": 5,
    "num_tasks": 40,
    "num_gpus": 1,
    "memory_gb": 8,
    "workdir": "",
}

args = task.connect(args)

#----------------------------------------------------------------------------------------------------
# Load bash script from file
#----------------------------------------------------------------------------------------------------
script = Path("gromacs_clearml_script.sh").read_text()

#----------------------------------------------------------------------------------------------------
# Modify bash script according to values provided from application wizard
#----------------------------------------------------------------------------------------------------
# MPI threads
script = re.sub(r"-ntmpi \d+", f"-ntmpi {args['num_mpi_threads']}", script, 1)

# OpenMP threads
script = re.sub(r"-ntomp \d+", f"-ntomp {args['num_openmp_threads']}", script, 1)

# GPUs
gpu_id_str = "".join(str(i) for i in range(args["num_gpus"]))
script = re.sub(r"-gpu_id \d+", f"-gpu_id {gpu_id_str}", script, 1)

# Working directory
if args["workdir"]:
    script = re.sub(r"WORKDIR=\"[^\"]*\"", f"WORKDIR={args['workdir']}", script, 1)

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
# Update num_tasks according to the number of MPI and OpenMP threads
#----------------------------------------------------------------------------------------------------
args["num_tasks"] = args["num_mpi_threads"] * args["num_openmp_threads"]

#----------------------------------------------------------------------------------------------------
# Update task hyperparameters according to application wizard (these will be used by the Slurm template)
#----------------------------------------------------------------------------------------------------
bash_task.set_parameters(
    {
        "properties/num_tasks": args["num_tasks"],
        "properties/num_gpus": args["num_gpus"],
        "properties/memory_gb": args["memory_gb"],
    }
)

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
