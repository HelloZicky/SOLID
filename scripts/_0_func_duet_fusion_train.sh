dataset=$1
model=$2
cuda_num=$3

ITERATION=24300
SNAPSHOT=2430
MAX_EPOCH=20
ARCH_CONF_FILE="configs_fusion/${dataset}_conf.json"


GRADIENT_CLIP=5                     # !
BATCH_SIZE=1024

######################################################################
LEARNING_RATE=0.001
CHECKPOINT_PATH=../checkpoint/MM2024/${dataset}_${model}/id_540
######################################################################

echo ${CHECKPOINT_PATH}
echo "Model save to ${CHECKPOINT_PATH}"


USER_DEFINED_ARGS="--model=meta_${model}_fusion --num_loading_workers=1 --arch_config=${ARCH_CONFIG_CONTENT} --learning_rate=${LEARNING_RATE} \
--max_gradient_norm=${GRADIENT_CLIP} --batch_size=${BATCH_SIZE} --snapshot=${SNAPSHOT} --max_steps=${ITERATION} --checkpoint_dir=${CHECKPOIN\
T_PATH} --arch_config=${ARCH_CONF_FILE}"

dataset="../../data/${dataset^}"

train_file="${dataset}/id/train.txt,${dataset}/id/540/${model}/train.txt"
test_file="${dataset}/id/test.txt,${dataset}/id/540/${model}/test.txt"
data="${train_file};${test_file}"

export CUDA_VISIBLE_DEVICES=${cuda_num}
echo ${USER_DEFINED_ARGS}
python ../main/multi_metric_meta_train_fusion.py \
--dataset=${data} \
${USER_DEFINED_ARGS}

echo "Training done: ${CHECKPOINT_PATH}"

