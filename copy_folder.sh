#!/bin/bash

# Set the source and destination paths
source_path="models"
destination_instance="instance-1" 
destination_project="apollosearch"

# Copy the models folder to the destination instance
gcloud compute scp --recurse "$source_path" "$destination_instance:~/ApolloSearchCore/models" --project "$destination_project"
