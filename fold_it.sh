#!/bin/bash

### EXPERIMENTO Y MODELOS SELECCIONADOS
EXPERIMENT_NAME="pseudop_0"
MODEL_NAME="unet"

# File paths
MODEL_PATH="src/seq2seq/model"
INIT="src/seq2seq/__init__.py"
GLOBAL_CONFIG="config/global.json" 
MODEL_FILE="$MODEL_PATH/$MODEL_NAME.py"

BASE_OUTPUT_PATH="results/test0/$EXPERIMENT_NAME"
mkdir -p "$BASE_OUTPUT_PATH"

# # Hyperparameters 
num_conv=(1 2)
n_4=(0 1 2)
n_8=(2 3)
skip=(0 1)

### LOGGING de la ejecución
cp fold_it.sh "$BASE_OUTPUT_PATH"
echo "# Starting experiment -- $EXPERIMENT_NAME -- at $(date)" > "$BASE_OUTPUT_PATH/models.log"
echo "Saving hyperparameters:" \
     "num_conv = ${num_conv[@]}," \
     "n_4 = ${n_4[@]}," \
     "n_8 = ${n_8[@]}," \
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
            for s in "${skip[@]}"; do
                echo "########################################################################"
                save_name="pseudop_nc_${nc}_n4_${nc_4}_n8_${nc_8}_skip${s}"
                echo "Ejecutando: $save_name"
                # # Modificar la configuración del modelo para pool, up y skip
                sed -i \
                    -e "83s/\(num_conv=\)[0-9e.-]*/num_conv=$nc/" \
                    -e "86s/\(skip=\)[0-9e.-]*/skip=$s/" \
                    -e "93s/n_4=[0-9e.-]*/n_4=$nc_4/" \
                    -e "94s/n_8=[0-9e.-]*/n_8=$nc_8/" \
                    "$MODEL_FILE"
                bash scripts/train_test.sh "$BASE_OUTPUT_PATH" "$save_name" 
            done
        done 
    done 
done

