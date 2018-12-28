#!/bin/bash
# nohup ./run.sh 0 1000 > YuCNN.log 2>&1 &
# tail -f YuCNN.log
# ./run.sh 0 1000 > CNN-GRNN.log 2>&1
# tail -f CNN-GRNN.log

source keras-env/bin/activate
#python make_dataset.py config.yaml
#python local_contrast_normalization.py
for ((i=$1; i<$2; i++)); do
    echo $i
    python split_train_test.py $i # EXP_ID
    python patch_dataset.py
    python cnn.py $i
done;
deactivate