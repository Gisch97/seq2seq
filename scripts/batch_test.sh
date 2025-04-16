#!/bin/bash

### EXPERIMENTO Y MODELOS SELECCIONADOS
# EXPERIMENT_NAME="test"
EXPERIMENT_NAME="paper_based_unet_batch_test"

GLOBAL_CONFIG="config/global.json" 
TEST_CONFIG="config/test.json"

BASE_OUTPUT_PATH="results/UNet_selection/paper_based_unet/"
LOAD=$BASE_OUTPUT_PATH'paper_based_unet_cdhit-skip0'
SAVE=$BASE_OUTPUT_PATH'paper_based_unet_batch_test'
mkdir -p "$SAVE" 
BATCHES=(160 192 224 256 288)

echo $TEST_CONFIG
for batch in {192..288..32}; do
    save_name="batch_size_"$batch
    echo "########################################################################"
    echo "Updating global configuration... "
    echo "run: $SAVE $batch"
    sed -i \
        -e "s/\"run\": \"[^\"]*\"/\"run\": \"batch-$batch\"/" \
        -e "s/\"batch_size\": [0-9e.-]*/\"batch_size\": $batch/" \
         "$GLOBAL_CONFIG"

    # Update test configuration
    echo "########################################################################"
    echo "Updating test configuration... " 
    sed -i \
        -e "s|\"out_path\": \"[^\"]*\"|\"out_path\": \"$SAVE/test_batch_$batch.csv\"|" \
        "$TEST_CONFIG"

    # Test model
    seq2seq test
    echo "Testing completed for configuration: $save_name"
done
