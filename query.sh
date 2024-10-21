#!/bin/bash

# Define the checkpoint directory
CHECKPOINT_DIR=~/.llama/checkpoints/Llama3.2-3B-Instruct

# # Prompt the user for a query
# echo "Please enter your query:"
# read USER_QUERY

# Set the PYTHONPATH and run the Python script with the user query
PYTHONPATH=$(git rev-parse --show-toplevel) torchrun models/scripts/example_chat_completion.py $CHECKPOINT_DIR
