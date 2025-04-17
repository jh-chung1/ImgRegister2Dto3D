#!/bin/bash

#SBATCH --mem=192GB -p serc 
#SBATCH --time=4:00:00
#SBATCH --job-name="c24"
#SBATCH --output=job_main_c24_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24  

echo "Job started at: $(date)"
echo "CPUs used: ${SLURM_CPUS_PER_TASK}"

START_TIME=$SECONDS

# real
./miniconda3/envs/ImgRegister2Dto3D/bin/python ../ImgReg_main.py --CT_data_dir '../data/B1M1_CT_binary_8.02um_8bu_748x748x1234.raw' --Template_dir '../data/B1M1_TS_binary_10X_PPL_8bu_13776_8460.raw' --test_type 'Real_BereaSandstone' --cpu_num 24 --section_axis 2

# val
./miniconda3/envs/ImgRegister2Dto3D/bin/python ../ImgReg_main.py --CT_data_dir '../data/SpherePack_300x300x300.raw' --Template_dir '../data/SpherePack_template_Rotation_X_5_Y_4_Z_-3_SectNo_155.raw' --test_type 'Validation_SpherePack' --cpu_num 24 --section_axis 1

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "Job ended at: $(date)"
echo "Total runtime: $(($ELAPSED_TIME / 3600))h $((($ELAPSED_TIME / 60) % 60))m $(($ELAPSED_TIME % 60))s"
