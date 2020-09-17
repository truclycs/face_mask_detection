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
# python3 apis/start_server.py

DIR=$(pwd)
FILE_START=$(find -type f -name "start_server.py")
START_PATH=$DIR${FILE_START:1}

#write out current crontab
crontab -l > crontab_run_server_FMD
#echo new cron into cron file
echo "0 5 1 * * python3 $START_PATH" >> crontab_run_server_FMD
# echo "*/1 * * * * python3 $START_PATH" >> crontab_run_server_FMD
#install new cron file
crontab crontab_run_server_FMD
rm crontab_run_server_FMD
crontab -l