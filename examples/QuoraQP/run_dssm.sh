#!/usr/bin/env bash

cd ../../

currpath=`pwd`
# train the model
python matchzoo/main.py --phase train --model_file ${currpath}/examples/QuoraQP/config/dssm_quoraqp.config

# predict with the model

python matchzoo/main.py --phase predict --model_file ${currpath}/examples/QuoraQP/config/dssm_quoraqp.config
