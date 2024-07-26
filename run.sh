#!/bin/bash
WORKING_PATH=$(pwd)
ModelDir=${WORKING_PATH}/model

# if you are using conda env, you can use command below to activate
# Conda_Path=path/to/conda/envs
# source ${Conda_Path}/bin/activate
# export PATH=${Conda_Path}/bin/:$PATH


scriptlist=("etr.py" "rfr.py" "svr.py")
for script in ${scriptlist[*]};
do
echo "$script start"

python $ModelDir/${script}

done