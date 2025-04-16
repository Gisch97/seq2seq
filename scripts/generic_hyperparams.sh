#!/bin/bash

# Configuración de hiperparámetros para las pruebas
latent_dims=(12 16 20 32 64)
negative_weights=(0.1 0.2 0.5)
learning_rates=(1e-4 1e-3 1e-2)
schedulers=("none" "cosine" "linear")
interaction_priors=("False" "True")
output_thresholds=(0.5 0.7)

# Directorios y archivos de configuración
model_file="src/seq2seq/model.py"
global_config="config/global.json"
train_config="config/train.json"

# Bucle para cada combinación de hiperparámetros
for dim in "${latent_dims[@]}"; do
  for neg_weight in "${negative_weights[@]}"; do
    for lr in "${learning_rates[@]}"; do
      for scheduler in "${schedulers[@]}"; do
        for interaction_prior in "${interaction_priors[@]}"; do
          for out_th in "${output_thresholds[@]}"; do
            echo "Ejecutando con latent_dim=$dim, negative_weight=$neg_weight, lr=$lr, scheduler=$scheduler, interaction_prior=$interaction_prior, output_th=$out_th"

            # Modificar el archivo model.py
            sed -i "s/latent_dim=[0-9]*/latent_dim=$dim/" "$model_file"
            sed -i "s/negative_weight=[0-9.]*,/negative_weight=$neg_weight,/" "$model_file"
            sed -i "s/lr=[0-9.e-]*/lr=$lr/" "$model_file"
            sed -i "s/scheduler=\"[a-z]*\"/scheduler=\"$scheduler\"/" "$model_file"
            sed -i "s/interaction_prior=[a-zA-Z]*/interaction_prior=$interaction_prior/" "$model_file"
            sed -i "s/output_th=[0-9.]*,/output_th=$out_th,/" "$model_file"

            # Modificar el archivo global.json
            sed -i "s/\"run\": \"[^\"]*\"/\"run\": \"latent$dim-neg$neg_weight-lr$lr-$scheduler-ip$interaction_prior-ot$out_th\"/" "$global_config"

            # Modificar el archivo train.json
            sed -i "s/latent-[0-9]*/latent-$dim/" "$train_config"

            # Ejecutar el entrenamiento
            seq2seq train

            echo "Finalizado con latent_dim=$dim, negative_weight=$neg_weight, lr=$lr, scheduler=$scheduler, interaction_prior=$interaction_prior, output_th=$out_th"
          done
        done
      done
    done
  done
done
