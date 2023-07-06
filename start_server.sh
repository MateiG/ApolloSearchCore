#!/bin/bash

cd ~/ApolloSearchCore/

source venv/bin/activate

export GOOGLE_APPLICATION_CREDENTIALS="credentials/apollosearch-bebb5b92d946.json"
export OPENAI_API_KEY="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"
export MIXPANEL_KEY="091f7f4f16d98b2155901f950b488c1b"

nohup gunicorn wsgi:app --workers 4 --bind 127.0.0.1:8000 --timeout 120  > output.log 2>&1 &

echo "Gunicorn server started"
