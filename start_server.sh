#!/bin/bash

cd ~/ApolloSearchCore/

source venv/bin/activate

export GOOGLE_APPLICATION_CREDENTIALS="credentials/apollosearch-bebb5b92d946.json"
export OPENAI_API_KEY="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"

nohup gunicorn -w 4 -b 127.0.0.1:8000 wsgi:app > output.log 2>&1 &

echo "Gunicorn server started"
