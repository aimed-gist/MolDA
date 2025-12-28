#!/bin/bash

# Target GPUs (0,1,2,3)
TARGET_GPUS="0 1 2 3"

echo "==============Killing Test Processes (GPU 0,1,2,3)==============="

# 1. Find and kill stage3.py processes (all test types)
echo "[1/4] Searching for stage3.py processes..."
pids=$(ps aux | grep -E "python.*stage3\.py" | grep -v grep | awk '{print $2}')

if [ -z "$pids" ]; then
    echo "  No stage3.py processes found."
else
    echo "  Found stage3.py processes: $pids"
    for pid in $pids; do
        echo "  Killing process $pid..."
        kill -9 $pid 2>/dev/null
    done
fi

# 2. Find and kill multiprocessing spawn workers
echo ""
echo "[2/4] Searching for multiprocessing worker processes..."
worker_pids=$(ps aux | grep -E "multiprocessing\.(spawn|forkserver|fork)" | grep -v grep | awk '{print $2}')

if [ -z "$worker_pids" ]; then
    echo "  No worker processes found."
else
    echo "  Found worker processes: $worker_pids"
    for pid in $worker_pids; do
        echo "  Killing worker $pid..."
        kill -9 $pid 2>/dev/null
    done
fi

# 3. Find and kill python processes with high CPU (>50%)
echo ""
echo "[3/4] Searching for high CPU python processes..."
high_cpu_pids=$(ps aux | grep python | grep -v grep | awk '$3 > 50.0 {print $2}')

if [ -z "$high_cpu_pids" ]; then
    echo "  No high CPU processes found."
else
    echo "  Found high CPU processes: $high_cpu_pids"
    for pid in $high_cpu_pids; do
        echo "  Killing high CPU process $pid..."
        kill -9 $pid 2>/dev/null
    done
fi

# 4. Find and kill GPU processes on target GPUs (0,1,2,3) only
echo ""
echo "[4/4] Searching for GPU processes on GPU 0,1,2,3..."
if command -v nvidia-smi &> /dev/null; then
    # Get PIDs for each target GPU
    for gpu_id in $TARGET_GPUS; do
        gpu_pids=$(nvidia-smi --id=$gpu_id --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
        if [ -n "$gpu_pids" ]; then
            echo "  GPU $gpu_id processes: $gpu_pids"
            for pid in $gpu_pids; do
                # Check if it's a python process before killing
                proc_name=$(ps -p $pid -o comm= 2>/dev/null)
                if [[ "$proc_name" == *"python"* ]]; then
                    echo "    Killing GPU $gpu_id process $pid ($proc_name)..."
                    kill -9 $pid 2>/dev/null
                fi
            done
        fi
    done
else
    echo "  nvidia-smi not available, skipping GPU process check."
fi

# Wait a moment for processes to terminate
sleep 1

echo ""
echo "==============Cleanup Complete==============="

# Show remaining python processes
echo ""
echo "Remaining python processes:"
remaining=$(ps aux | grep python | grep -v grep | grep -v "ipykernel" | grep -v "jupyter" | grep -v "vscode")
if [ -z "$remaining" ]; then
    echo "  (none)"
else
    echo "$remaining" | head -10
fi

# Show GPU memory status for target GPUs
echo ""
echo "GPU Memory Status (GPU 0,1,2,3):"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --id=0,1,2,3 --query-gpu=index,memory.used,memory.total --format=csv 2>/dev/null || echo "  Could not query GPU status."
else
    echo "  nvidia-smi not available."
fi

echo ""
echo "If you still see running processes, you can manually kill them with:"
echo "  kill -9 <PID>"
