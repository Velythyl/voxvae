#!/bin/bash
#SBATCH --job-name=vma_job          # Job name
#SBATCH --nodes=1                   # Use 1 node
#SBATCH --ntasks-per-node=1         # Run a single task
#SBATCH --cpus-per-task=30           # Number of CPU cores per task
#SBATCH --gres=gpu:rtx8000:1                # Number of GPUs per node
#SBATCH --time=4-00:00:00           # Time limit (4 days)
#SBATCH --output=vma_%j.out         # Output file
#SBATCH --error=vma_%j.err          # Error file
#SBATCH --mem=40G                   # Memory request (adjust as needed)



# Activate your virtual environment
source venv/bin/activate

# Run your Python script
python3 voxvae/vma.py
