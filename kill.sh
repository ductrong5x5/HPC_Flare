#!/bin/bash

# Prompt the user to enter the process name
process_name="nvflare"
# process_name=fedmetrix
# Kill the bash processes
bash_pids=$(ps aux | grep "[b]ash.*$process_name" | awk '{print $2}')
if [ -z "$bash_pids" ]; then
    echo "No bash process found with name: $process_name"
else
    echo "Killing bash processes with name: $process_name"
    echo "$bash_pids" | xargs kill -9
    echo "Killed bash processes: $bash_pids"
fi

# Kill the python3 processes
python3_pids=$(ps aux | grep "[p]ython3.*$process_name" | awk '{print $2}')
if [ -z "$python3_pids" ]; then
    echo "No python3 process found with name: $process_name"
else
    echo "Killing python3 processes with name: $process_name"
    echo "$python3_pids" | xargs kill -9
    echo "Killed python3 processes: $python3_pids"
fi