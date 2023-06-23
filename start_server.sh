#!/bin/bash

cd ~/ApolloSearchCore/

source venv/bin/activate

nohup gunicorn -w 4 -b 127.0.0.1:8000 wsgi:app > output.log 2>&1 &

echo "Gunicorn server started"
