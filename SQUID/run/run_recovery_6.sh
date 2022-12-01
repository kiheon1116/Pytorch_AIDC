#!/bin/bash

# $1 : gpu device num
# $2 : nbit
MAIN_PY="../srcs/main.py"
iteration=10
iteration_end=`expr 100 - ${iteration}`
weight="IMAGENET1K_V1"

for ((start=0; start<=${iteration_end}; start+=${iteration}))
do
    for model in resnet50 densenet169 inception_v3 mobilenet_v2 vgg19 regnet_x_8gf efficientnet_b2 convnext_base
    do
        ber=`expr $1 - 5`
        CUDA_VISIBLE_DEVICES=$1 python ${MAIN_PY} --model ${model} --weight ${weight} --ber ${ber} --ber_stride 10 --iteration ${iteration} --iteration_start ${start} --g_seed 0 --nbit 6 --option squid --quant channel
    done
done