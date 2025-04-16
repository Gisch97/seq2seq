#!/bin/bash

### EXPERIMENTO Y MODELOS SELECCIONADOS
EXPERIMENT_NAME="Unet_paper_noise_0_1"
MODEL_NAME="unet_original" 

# File paths
MODEL_PATH="src/seq2seq/models/paper_based"
INIT="src/seq2seq/__init__.py"
GLOBAL_CONFIG="config/global.json" 
MODEL_FILE="$MODEL_PATH/$MODEL_NAME.py"

BASE_OUTPUT_PATH="results/UNet_selection/$EXPERIMENT_NAME"
mkdir -p "$BASE_OUTPUT_PATH"

# Hyperparameters
MAX_EPOCHS=20
pool_mode=('avg')
up_mode=('transpose')
addition=('sum')
skip=(0 1)
num_conv=(1 2)

n_4=(0 1 2)
n_8=(0 1 2)
         

### LOGGING de la ejecución
cp fold_it.sh "$BASE_OUTPUT_PATH"
echo "# Starting experiment -- $EXPERIMENT_NAME -- at $(date)" > "$BASE_OUTPUT_PATH/models.log"
echo "Saving hyperparameters:" \
     "pool mode = ${pool_mode[@]}," \
     "up mode = ${up_mode[@]}," \
     "addition = ${addition[@]}," \
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
sed -i "s/\"max_epochs\": [0-9e.-]*/\"max_epochs\": $MAX_EPOCHS/" "$GLOBAL_CONFIG"

# Bucle para iterar sobre las configuraciones de pool, up y addition
for nc in "${num_conv[@]}"; do
    for nc_4 in "${n_4[@]}"; do
        for nc_8 in "${n_8[@]}"; do
            # Se configura el valor de skip fijo en 0 para este bloque (se puede anidar otro bucle si se desea variar)
            base_name="nc-$nc-n4-$nc_4-n8-$nc_8"
            echo "Ejecutando: -n4-$nc_4-n8-$nc_8 skip=0"
            save_name="${base_name}-skip0"
            # # Modificar la configuración del modelo para pool, up y skip
            sed -i \
                -e "87s/\(skip=\)[0-9e.-]*/skip=0/" \
                -e "84s/\(num_conv=\)[0-9e.-]*/num_conv=$nc/" \
                -e "94s/n_4=[0-9e.-]*/n_4=$nc_4/" \
                -e "95s/n_8=[0-9e.-]*/n_8=$nc_8/" \
                "$MODEL_FILE"

            bash scripts/train.sh "$BASE_OUTPUT_PATH" "$save_name" 
            for swaps in {30..0..-2}; do
                bash scripts/test_w_swap.sh "$BASE_OUTPUT_PATH" "$save_name" "$swaps"
            done

            echo "Ejecutando: -n4-$nc_4-n8-$nc_8 skip=1"
            # Actualizar el parámetro addition en el modelo
            sed -i \
                -e "87s/\(skip=\)[0-9e.-]*/skip=1/" \
                "$MODEL_FILE"
            # Definir el nombre de guardado usando base_name y el valor de addition
            save_name="$base_name-skip1"
            echo "save_name: $save_name"

            bash scripts/train.sh "$BASE_OUTPUT_PATH" "$save_name" 
            for swaps in {30..0..-2}; do
                bash scripts/test_w_swap.sh "$BASE_OUTPUT_PATH" "$save_name" "$swaps"
            done
        
        done
    done 
done 

