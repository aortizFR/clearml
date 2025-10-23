#!/bin/bash

# Apptainer image path
export APPTAINER_IMAGE="${HOME}/clearml/gromacs/gromacs_2023.2.sif"

# Gromacs working directory
export WORKDIR="${HOME}/clearml/gromacs/stmv_benchmark/GROMACS_heterogeneous_parallelization_benchmark_info_and_systems_JCP/stmv"

# Gromacs variables
export GMX_ENABLE_DIRECT_GPU_COMM=1

apptainer exec --nv ${APPTAINER_IMAGE} nvidia-smi

cd ${WORKDIR}
apptainer run --nv \
       --bind ${HOME}:${HOME}:rw \
       ${APPTAINER_IMAGE} \
       gmx mdrun -ntmpi 8 -ntomp 5 -nb gpu -pme gpu -npme 1 -update gpu -bonded gpu -nsteps 100000 -resetstep 90000 -noconfout -dlb no -nstlist 300 -pin on -v -gpu_id 0

exit 0
