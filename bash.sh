#!/bin/bash
Default_folder_name=/home/kkh/pytorch
folder_name=param_comp_data

for name in ${Default_folder_name}/data/${folder_name}/*weight.npy
do
    python Extract.py --filename ${name};
done
cd ${Default_folder_name}/data/${folder_name}/
mv *.png ${Default_folder_name}/picture/${folder_name}/
