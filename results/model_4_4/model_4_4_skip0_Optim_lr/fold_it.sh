#!/bin/bash

### EXPERIMENTO Y MODELOS SELECCIONADOS
EXPERIMENT_NAME="model_4_4_skip0_Optim_lr"
MODEL_NAME="unet" 

# File paths
MODEL_PATH="src/seq2seq/model/"
INIT="src/seq2seq/__init__.py"
GLOBAL_CONFIG="config/global.json" 
MODEL_FILE="$MODEL_PATH/$MODEL_NAME.py"

BASE_OUTPUT_PATH="results/model_4_4/$EXPERIMENT_NAME"
mkdir -p "$BASE_OUTPUT_PATH"

# Hyperparameters 
pool_mode=('avg')
up_mode=('transpose')
addition=('sum')
skip=(0)
n_4=(1)
num_conv=(1 2) 
learning_rates=(1e-3 3e-3 5e-3 7e-3 9e-3 1e-2 3e-2 5e-2 7e-2 9e-2 1e-1 3e-1)

### LOGGING de la ejecución
cp fold_it.sh "$BASE_OUTPUT_PATH"
echo "# Starting experiment -- $EXPERIMENT_NAME -- at $(date)" > "$BASE_OUTPUT_PATH/models.log"
echo "Saving hyperparameters:" \
     "learning_rates = ${learning_rates[@]}," \
     "num_conv = ${num_conv[@]}," | tee -a "$BASE_OUTPUT_PATH/models.log"
 
echo "# Config train #" >> "$BASE_OUTPUT_PATH/models.log"
cat config/train.json >> "$BASE_OUTPUT_PATH/models.log"
echo "# Config test #" >> "$BASE_OUTPUT_PATH/models.log"
cat config/test.json >> "$BASE_OUTPUT_PATH/models.log"

echo "# Guardando contenido del archivo del modelo: $MODEL_FILE" >> "$BASE_OUTPUT_PATH/models.log"
echo "# Contenido del modelo:" >> "$BASE_OUTPUT_PATH/models.log"
cat "$MODEL_FILE" >> "$BASE_OUTPUT_PATH/models.log"

# Actualiza el archivo de configuración global
sed -i "s/\"exp\": \"[^\"]*\"/\"exp\": \"$EXPERIMENT_NAME\"/" "$GLOBAL_CONFIG"

# Bucle para iterar sobre las configuraciones de pool, up y addition

for nc in "${num_conv[@]}"; do 
    for lr in "${learning_rates[@]}"; do
        for iter in {1..5}; do
            # Se configura el valor de skip fijo en 0 para este bloque (se puede anidar otro bucle si se desea variar)
            base_name="nc$nc-lr-$lr-iter$iter"
            echo "Ejecutando: nc=$nc-lr-$lr iter $iter"
            save_name="${base_name}"
            # # Modificar la configuración del modelo para pool, up y skip
            sed -i \
                -e "37s/lr=[0-9.e-]*/lr=$lr/" "$model_file"\
                -e "83s/\(num_conv=\)[0-9e.-]*/num_conv=$nc/" \
                "$MODEL_FILE"
            bash scripts/train_test.sh "$BASE_OUTPUT_PATH" "$save_name"
        done
    
    done 
done 

