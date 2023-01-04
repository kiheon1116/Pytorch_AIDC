#!/bin/bash

# $1 : gpu device num
# $2 : nbit
MAIN_PY="../AEC/main.py"
iteration=10
iteration_end=`expr 50 - ${iteration}`
weight="IMAGENET1K_V1"
ber=`expr $1 - 5`

for ((start=0; start<=${iteration_end}; start+=${iteration}))
do
    for model in inception_v3 mobilenet_v2 regnet_x_8gf efficientnet_b2
    do        
        CUDA_VISIBLE_DEVICES=$1 python ${MAIN_PY} --model ${model} --weight ${weight} --ber ${ber} --ber_stride 4 --iteration ${iteration} --iteration_start ${start} --g_seed 0 --nbit 1 --option weight_nulling --quant layer 
    done
done
