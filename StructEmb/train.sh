#!/bin/bash

# Parameter settings
configs=(
    "1 2560 1e-4 30 0.01 0.3 0.005 0.002 2 0.012 200 random add",
)

# DistributedDataParallel training
for config in "${configs[@]}"; do
    echo "Running experiment with config: $config"
    python main_ddp.py --run_number $(echo $config | awk '{print $1}') \
                       --bs $(echo $config | awk '{print $2}') \
                       --lr $(echo $config | awk '{print $3}') \
                       --epoch $(echo $config | awk '{print $4}') \
                       --aug_p1 $(echo $config | awk '{print $5}') \
                       --aug_p3_1 $(echo $config | awk '{print $6}') \
                       --aug_p3_2 $(echo $config | awk '{print $7}') \
                       --aug_p3_3 $(echo $config | awk '{print $8}') \
                       --num_layers $(echo $config | awk '{print $9}') \
                       --tau $(echo $config | awk '{print $10}') \
                       --hidden_dim $(echo $config | awk '{print $11}') \
                       --initial_node $(echo $config | awk '{print $12}') \
                       --pool_mode $(echo $config | awk '{print $13}')
done
