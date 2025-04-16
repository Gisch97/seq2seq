#!/bin/bash
#   train_test.sh  
#   Este script sirve para ejecutar entrenamientos y test de modelos, guardandolos con mlflow.
#
GLOBAL_CONFIG="config/global.json"
TRAIN_CONFIG="config/train.json"
 
BASE_OUTPUT_PATH=$1
save_name=$2
 
save_path="$BASE_OUTPUT_PATH/$save_name"

echo "########################################################################"
echo "Updating global configuration... "
echo "run: $save_name"
sed -i "s/\"run\": \"[^\"]*\"/\"run\": \"$save_name\"/" "$GLOBAL_CONFIG"

# Update train configuration

echo "########################################################################"
echo "Updating train configuration..."
echo "out_path: $save_name"
sed -i "s|\"out_path\": \"[^\"]*\"|\"out_path\": \"$save_path\"|" "$TRAIN_CONFIG"

# Train model

echo "########################################################################"
seq2seq train
echo "Training completed for configuration: $save_name"