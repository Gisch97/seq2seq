#!/bin/bash
#   train_test.sh  
#   Este script sirve para ejecutar entrenamientos y test de modelos, guardandolos con mlflow.
#
GLOBAL_CONFIG="config/global.json"
TRAIN_CONFIG="config/train.json"
TEST_CONFIG="config/test.json"
 
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

# Update test configuration
echo "########################################################################"
echo "Updating test configuration... "
echo "out_path: $save_name"
sed -i \
    -e "s|\"model_weights\": \"[^\"]*\"|\"model_weights\": \"$save_path/weights.pmt\"|" \
    -e "s|\"out_path\": \"[^\"]*\"|\"out_path\": \"$save_path/test.csv\"|" \
    "$TEST_CONFIG"

# Test model
seq2seq test
echo "Testing completed for configuration: $save_name"