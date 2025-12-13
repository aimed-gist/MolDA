#!/bin/bash

echo "==============Killing Galactica Test Processes==============="

# Find and kill stage3.py processes
echo "Searching for stage3.py processes..."
pids=$(ps aux | grep "stage3.py" | grep "test_galactica" | grep -v grep | awk '{print $2}')

if [ -z "$pids" ]; then
    echo "No stage3.py processes found."
else
    echo "Found stage3.py processes: $pids"
    for pid in $pids; do
        echo "Killing process $pid..."
        kill -9 $pid 2>/dev/null
    done
fi

# Find and kill all child python processes (multiprocessing workers)
echo ""
echo "Searching for multiprocessing worker processes..."
worker_pids=$(ps aux | grep "multiprocessing.spawn" | grep "mol_llm_env" | grep -v grep | awk '{print $2}')

if [ -z "$worker_pids" ]; then
    echo "No worker processes found."
else
    echo "Found worker processes: $worker_pids"
    for pid in $worker_pids; do
        echo "Killing worker $pid..."
        kill -9 $pid 2>/dev/null
    done
fi

# Find and kill any remaining python processes running in mol_llm_env with high CPU
echo ""
echo "Searching for high CPU python processes in mol_llm_env..."
high_cpu_pids=$(ps aux | grep "mol_llm_env/bin/python" | awk '$3 > 50.0 {print $2}' | grep -v grep)

if [ -z "$high_cpu_pids" ]; then
    echo "No high CPU processes found."
else
    echo "Found high CPU processes: $high_cpu_pids"
    for pid in $high_cpu_pids; do
        echo "Killing high CPU process $pid..."
        kill -9 $pid 2>/dev/null
    done
fi

echo ""
echo "==============Cleanup Complete==============="
echo "Remaining python processes in mol_llm_env:"
ps aux | grep "mol_llm_env/bin/python" | grep -v grep | grep -v "ipykernel" | head -5

echo ""
echo "If you still see running processes, you can manually kill them with:"
echo "kill -9 <PID>"
