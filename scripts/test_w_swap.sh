#!/bin/bash
#   train_test.sh  
#   Este script sirve para ejecutar entrenamientos y test de modelos, guardandolos con mlflow.
#
GLOBAL_CONFIG="config/global.json"
TRAIN_CONFIG="config/train.json"
TEST_CONFIG="config/test.json"
 
BASE_OUTPUT_PATH=$1
save_name=$2
swaps=$3

save_path="$BASE_OUTPUT_PATH/$save_name"

echo "Updating global configuration... "
echo "run: $save_name-$swaps"
sed -i \
    -e "s/\"run\": \"[^\"]*\"/\"run\": \"$save_name-$swaps\"/" \
    -e "s/\"n_swaps\": [0-9e.-]*/\"n_swaps\": $swaps/" \
        "$GLOBAL_CONFIG"


# Update test configuration
echo "Updating test configuration... "
echo "out_path: $save_name"
sed -i \
    -e "s|\"model_weights\": \"[^\"]*\"|\"model_weights\": \"$save_path/weights.pmt\"|" \
    -e "s|\"out_path\": \"[^\"]*\"|\"out_path\": \"$save_path/test_n_swaps$swaps.csv\"|" \
    "$TEST_CONFIG"

# Test model
seq2seq test
echo "Testing completed for configuration: $save_name $swaps"