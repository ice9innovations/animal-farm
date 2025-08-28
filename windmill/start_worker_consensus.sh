#!/bin/bash
python consensus_worker.py > logs/consensus_worker.log 2>&1 &
echo $! > pids/consensus_worker.pid
echo "Started consensus worker (PID: $!)"
