#!/usr/bin/env bash
source ~/.bashrc
export PATH=~/anaconda3/bin:$PATH

#reads input from command line into a specific variable
read -p "Create new conda env (y/n)?" NEWENV
conda init bash

if [ "$NEWENV" == "y" ]; then
    # user chooses to create conda env
    # prompt user for conda env name
    echo "Creating new conda environment named monitoring-dashboard"
    # create conda environment with .yaml file
    conda env create -f environment.yml

else
    echo "No new environment created"
fi

echo "You may need to Stop and Relaunch My Server in Jupyter Hub if you use conda activate for the first time!"
