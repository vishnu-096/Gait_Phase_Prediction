#!/bin/bash

# Set the maximum number of restarts
MAX_RESTARTS=12
# Set the path to your Python script
PYTHON_SCRIPT="vae_ver5.py"

PICKLE_FILE="sub_cnt.pkl"

# Function to initialize the counter in the pickle file
initialize_counter() {
    python3 - <<END
import pickle

cnt = 0
with open('${PICKLE_FILE}', 'wb') as file:
    pickle.dump(cnt, file)
END
}

# Function to run the Python script
run_script() {
    python3 "${PYTHON_SCRIPT}"
}

initialize_counter
# Main loop
for ((i=1; i<=MAX_RESTARTS; i++)); do
    echo "Restarting Python script (Attempt $i)"
    
    # Run the Python script
    run_script

    # Check the exit status
    EXIT_STATUS=$?
    
    # Check if the script was killed by the OS (SIGKILL)
    if [ ${EXIT_STATUS} -eq 137 ]; then
        echo "Script was killed by the OS. Restarting..."
    else
        # Exit the loop if the script exited successfully
        echo "Script exited successfully."
        break
    fi

    # Sleep for a short duration before the next restart
    sleep 1
done

echo "Maximum number of restarts reached."
