#!/bin/bash
# A fake build script that runs for a short time
# and creates some child processes for the monitor to find.

echo "Fake build process started..."
echo "PID of this script: $$"

# Simulate some work and child processes
(sleep 1; echo "Fake child 1 ($BASHPID) finished.") &
(sleep 2; echo "Fake child 2 ($BASHPID) finished.") &

echo "Main fake build work (sleeping 3s)..."
sleep 3
wait # Wait for background children to complete

echo "Fake build process finished."
exit 0