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
EXPERIMENT_NAME="paper_based_unet-ch"
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
pool_mode=('max' 'avg') 
up_mode=('upsample' 'transpose')   
        
### LOGGING de la ejecución
cp fold_it.sh "$BASE_OUTPUT_PATH"
echo "# Starting experiment -- $EXPERIMENT_NAME -- at $(date)" > "$BASE_OUTPUT_PATH/models.log"
echo "Saving hyperparameters:" \
     "pool mode = ${pool_mode[@]}," \
     "up mode = ${up_mode[@]}," \
     "addition = ${addition[@]}," \
     "skip = ${skip[@]}" | tee -a "$BASE_OUTPUT_PATH/models.log"
echo "# Guardando contenido del archivo del modelo: $MODEL_FILE" >> "$BASE_OUTPUT_PATH/models.log"
echo "# Contenido del modelo:" >> "$BASE_OUTPUT_PATH/models.log"
cat "$MODEL_FILE" >> "$BASE_OUTPUT_PATH/models.log"

# Actualiza el archivo de configuración global
sed -i "s/\"exp\": \"[^\"]*\"/\"exp\": \"$EXPERIMENT_NAME\"/" "$GLOBAL_CONFIG"
sed -i "s/\"max_epochs\": [0-9e.-]*/\"max_epochs\": $MAX_EPOCHS/" "$GLOBAL_CONFIG"

# Bucle para iterar sobre las configuraciones de pool, up y addition
for p in "${pool_mode[@]}"; do
    for u in "${up_mode[@]}"; do

        # Se configura el valor de skip fijo en 0 para este bloque (se puede anidar otro bucle si se desea variar)
        base_name="unet-4-8-8-8-8-pool-$p-up-$u"
        echo "Ejecutando: pool=$p, up=$u, skip=0"
            save_name="${base_name}-skip1-add-cat"
        # Modificar la configuración del modelo para pool, up y skip
        sed -i \
            -e "81s/\(pool_mode=\)['\"][^'\"]*['\"]/pool_mode='$p'/" \
            -e "82s/\(up_mode=\)['\"][^'\"]*['\"]/up_mode='$u'/" \
            "$MODEL_FILE"

        echo "save_name: $save_name"
        bash scripts/train_test.sh "$BASE_OUTPUT_PATH" "$save_name" 
    done 
done 

