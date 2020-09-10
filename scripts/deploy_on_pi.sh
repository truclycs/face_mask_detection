#!/bin/bash

# Check virtual environment exist
if [ ! -d "venv" ]
then
    printf "[ERROR]: Virtual environment is not ready. 
            Please create virtual environment by run \'bash scripts/venv.sh\'\n"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate
printf "[INFO]: Activated virutal environment\n"

# Run API server
printf "[INFO]: API server is running\n"
# python3 streaming/run.py

DIR=$(pwd)
PYTHON=$(find -type f -name "python3")
FILE_START=$(find -type f -name "run.py")
PYTHON_PATH=$DIR${PYTHON:1}
START_PATH=$DIR${FILE_START:1}

#write out current crontab
crontab -l > crontab_run_server_FMD
#echo new cron into cron file
echo "0 5 1 * * $PYTHON_PATH $START_PATH" >> crontab_run_server_FMD
# echo "*/1 * * * * $PYTHON_PATH $START_PATH" >> crontab_run_server_FMD
#install new cron file
crontab crontab_run_server_FMD
rm crontab_run_server_FMD
crontab -l