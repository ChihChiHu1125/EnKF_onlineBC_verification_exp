#!/bin/bash
#PBS -N mpasjedi_var
#PBS -o mpasjedi_var.out
#PBS -e mpasjedi_var.err
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -k eod
#PBS -P 17001770
#PBS -m ae
#PBS -l select=1:ncpus=128:mpiprocs=128:mem=660GB

#export da_test="3denvar"
#export da_test="3denvar_dropout"
#export da_test="3denvar_single_obs"
#export da_test="3denvar_single_obs_nobc"
export da_test="3denvar_cloudpred"

export mpasjedi_dir=/home/project/17001770/weather_department/dae/hucc/MPAS_JEDI/build_ensembleonlinebc_260916/
#export mpasjedi_dir=/home/project/17001770/weather_department/dae/hucc/MPAS_JEDI/build_test_for_rttov_260714/
#export mpasjedi_dir=/home/project/17001770/weather_department/dae/hucc/MPAS_JEDI/new_build_260401/
#export mpasjedi_dir=/home/project/13004327/training_service/2026-MPAS-JEDI/SINGV-NG-1.0
#export mpasjedi_dir=/home/project/13004327/training_service/2026-MPAS-JEDI/MPAS-JEDI

#Env setup
source ${mpasjedi_dir}/code/env-setup/gnu-aspire2a.sh
export LD_LIBRARY_PATH=${mpasjedi_dir}/build/lib:$LD_LIBRARY_PATH

ulimit -s unlimited
export OOPS_TRACE=0
export OOPS_DEBUG=0
export F_UFMTENDIAN='big:101-200'
export GFORTRAN_CONVERT_UNIT='big_endian:101-200'
export OMP_NUM_THREADS=1
export FI_CXI_RX_MATCH_MODE='hybrid'

# Total MPI processes
ncpus=$(wc -l < $PBS_NODEFILE)

# Run executable
cd ${PBS_O_WORKDIR}
ln -sf ${mpasjedi_dir}/build/bin/mpasjedi_variational.x .
mpiexec -n $ncpus ./mpasjedi_variational.x ${da_test}.yaml jedi_${da_test}.log > jedi_${da_test}.log.all 2>&1

# Check for success
if ! grep -q 'Run: Finishing oops.* with status = 0' jedi_${da_test}.log; then
    echo "ERROR in $0 : jedi application failed" > ./FAIL
    exit 1
else
    rm jedi_${da_test}.log.0*
fi

exit 0
