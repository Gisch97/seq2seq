#!/bin/bash
# Pasos para ejecutar un entrenamiento:
#   1) Modificar la ruta de MODEL_FILE
#   2) Modificar en __init__.py el nombre del modelo importado
#   3) Modificar en GLOBAL_JSON el nombre del experimento
#   4) Modifiar los BASE_OUTPUT_PATH y save_name a utilizar con la nomenclatura correcta. (check skip / no-skip)
#   5) Modifiar los parametros a utilizar y bucles
#   6) modificar lineas 26 y 27 (log de hiperparams) 


### EXPERIMENTO Y MODELOS SELECCIONADOS
# EXPERIMENT_NAME="test"
EXPERIMENT_NAME="paper_based_unet_nc2_features"
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
# pool_mode=('max' 'avg') 
# up_mode=('upsample' 'transpose')  
# addition=('cat' 'sum')

pool_mode=('max') 
up_mode=('upsample')  
addition=('cat')
skip=(0 1)

n_8=(0 1 2 3 4)
n_16=(0 1 2 3)
        
# Bucle para iterar sobre las configuraciones de pool, up y addition
for nc_8 in "${n_8[@]}"; do
    for nc_16 in "${n_16[@]}"; do
        for p in "${pool_mode[@]}"; do
            for u in "${up_mode[@]}"; do
                # Se configura el valor de skip fijo en 0 para este bloque (se puede anidar otro bucle si se desea variar)
                base_name="unet_features-n8-$nc_8-n16-$nc_16-pool-$p-up-$u"
                echo "Ejecutando: pool=$p, up=$u, skip=0"
                save_name="${base_name}-skip0"
                echo "save_name: $save_name"

                for a in "${addition[@]}"; do 

                    echo "Ejecutando: pool=$p, up=$u, skip=0"
                    # Actualizar el parámetro addition en el modelo
                    # Definir el nombre de guardado usando base_name y el valor de addition
                    save_name="${base_name}-skip1-add-$a"
                    echo "save_name: $save_name"
                done
            done
        done
    done 
done 

