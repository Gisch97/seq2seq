#!/bin/bash

### EXPERIMENTO Y MODELOS SELECCIONADOS
EXPERIMENT_NAME="Train_w_noise_0_1"
MODEL_NAME="unet"

# File paths
MODEL_PATH="src/seq2seq/model"
INIT="src/seq2seq/__init__.py"
GLOBAL_CONFIG="config/global.json" 
MODEL_FILE="$MODEL_PATH/$MODEL_NAME.py"

BASE_OUTPUT_PATH="results/FlatNoise/$EXPERIMENT_NAME"
mkdir -p "$BASE_OUTPUT_PATH"

# Hyperparameters 
num_conv=(1 2)
n_8=(0 1)
n_4=(0 1 2 3)
train_swaps=(0.55 0.65 0.75 0.85 0.95)
skip=(0 1)

### LOGGING de la ejecución
cp fold_it.sh "$BASE_OUTPUT_PATH"
echo "# Starting experiment -- $EXPERIMENT_NAME -- at $(date)" > "$BASE_OUTPUT_PATH/models.log"
echo "Saving hyperparameters:" \
     "num_conv = ${num_conv[@]}," \
     "n_4 = ${n_4[@]}," \
     "n_8 = ${n_8[@]}," \
     "train_swaps = ${train_swaps[@]}," \
     "skip = ${skip[@]}" | tee -a "$BASE_OUTPUT_PATH/models.log"
 
echo "# Config train #" >> "$BASE_OUTPUT_PATH/models.log"
cat config/train.json >> "$BASE_OUTPUT_PATH/models.log"
echo "# Config test #" >> "$BASE_OUTPUT_PATH/models.log"
cat config/test.json >> "$BASE_OUTPUT_PATH/models.log"

echo "# Guardando contenido del archivo del modelo: $MODEL_FILE" >> "$BASE_OUTPUT_PATH/models.log"
echo "# Contenido del modelo:" >> "$BASE_OUTPUT_PATH/models.log"
cat "$MODEL_FILE" >> "$BASE_OUTPUT_PATH/models.log"

# Actualiza el archivo de configuración global
sed -i "s/\"exp\": \"[^\"]*\"/\"exp\": \"$EXPERIMENT_NAME\"/" "$GLOBAL_CONFIG"

# Bucle para iterar sobre las configuraciones
for nc in "${num_conv[@]}"; do
    for nc_8 in "${n_8[@]}"; do
        for nc_4 in "${n_4[@]}"; do
            for swaps in "${train_swaps[@]}"; do
                for s in "${skip[@]}"; do
                    echo "########################################################################"
                    save_name="nc_${nc}_n4_${nc_4}_n8_${nc_8}_skip${s}_noise${swaps}"
                    echo "Ejecutando: $save_name"
                    # # Modificar la configuración del modelo para pool, up y skip
                    sed -i \
                        -e "83s/\(num_conv=\)[0-9e.-]*/num_conv=$nc/" \
                        -e "86s/\(skip=\)[0-9e.-]*/skip=$s/" \
                        -e "93s/n_4=[0-9e.-]*/n_4=$nc_4/" \
                        -e "94s/n_8=[0-9e.-]*/n_8=$nc_8/" \
                        "$MODEL_FILE"
                    bash scripts/train_w_noise.sh "$BASE_OUTPUT_PATH" "$save_name" "$swaps"

                    echo "########################################################################"
                    echo " Testando modelo: $save_name"
                    bash scripts/test.sh "$BASE_OUTPUT_PATH" "$save_name" 
                done
            
            done
        done 
    done 
done

