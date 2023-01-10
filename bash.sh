#!/bin/bash
Default_folder_name=/home/kkh/pytorch
folder_name=param_INT10

for name in ${Default_folder_name}/data/${folder_name}/*weight.npy
do
    python Extract.py --filename ${name};
done
cd ${Default_folder_name}/data/
if[!-d ${folder_name}]; then
    cd ${Default_folder_name}/picture/
    mkdir ${folder_name} 
cd ${Default_folder_name}/data/${folder_name}/
mv *.png ${Default_folder_name}/picture/${folder_name}/
