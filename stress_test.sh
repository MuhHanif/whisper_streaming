#!/bin/bash

# Define the command you want to run
command_to_run="your_command_here"

# Define the number of times you want to run the command
num_runs=30

# Run the command multiple times in parallel
for ((i=1; i<=num_runs; i++)); do
  python3 whisper_online.py 1694630103.436884_msg0038.wav --language en --min-chunk-size 4 > out$i.txt &
done

# Wait for all background processes to finish
wait
