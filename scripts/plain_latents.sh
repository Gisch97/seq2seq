#!/bin/bash

# Directorios y archivos de configuración
model_file="src/seq2seq/model.py"
global_config="config/global.json"
train_config="config/train.json"
test_config="config/test.json"

# Configuración de hiperparámetros
l2_lambdas=(1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1) 

# Bucle para cada combinación de hiperparámetros
for l2 in "${l2_lambdas[@]}"; do 
    echo "Ejecutando con lambda=$l2"

    # Modificar el archivo model.py 
    sed -i "43s/ lambda_l2=[0-9e.-]*/lambda_l2=$l2/" "$model_file"

    # Modificar el archivo global.json
    sed -i "s/\"run\": \"[^\"]*\"/\"run\": \"loss_l2_$l2\"/" "$global_config"

    # Modificar el archivo train.json
    sed -i "s|\"out_path\": \"[^\"]*\"|\"out_path\": \"results/plain_loss_l2/lambda_$l2\"|" "$train_config"


    # Ejecutar el entrenamiento
    seq2seq train
    echo "Entrenamiento Finalizado (plain L2 loss) con lambda=$l2"

    # Modificar el archivo test.json
    sed -i "s|\"model_weights\": \"[^\"]*\"|\"model_weights\": \"results/plain_loss_l2/lambda_$l2/weights.pmt\"|" "$test_config"
    sed -i "s|\"out_path\": \"[^\"]*\"|\"out_path\": \"results/plain_loss_l2/lambda_$l2/test.csv\"|" "$test_config"
    

    # Ejecutar el entrenamiento
    seq2seq test
    echo "Prueba Finalizada (plain L2 loss) con lambda=$l2"
done