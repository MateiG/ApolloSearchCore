#!/bin/bash

# Find the Gunicorn process ID
PID=$(ps aux | grep wsgi:app | awk '{print $2}')

# Terminate the Gunicorn process
if [[ -n $PID ]]; then
    kill $PID
    echo "Gunicorn server stopped."
else
    echo "Gunicorn server is not running."
fi
