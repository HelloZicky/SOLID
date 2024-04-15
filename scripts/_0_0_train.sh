#!/bin/bash
date
#dataset_list=("amazon_beauty" "amazon_cds" "amazon_electronic")
dataset_list=("amazon_arts_subset")
# dataset_list=("amazon_instruments_subset")
# dataset_list=("amazon_office_subset")
# dataset_list=("amazon_scientific_subset")
# dataset_list=("amazon_cds")
# dataset_list=("amazon_electronic" "amazon_cds")
# dataset_list=("douban_book" "douban_music")
# dataset_list=("douban_music")
# dataset_list=("movielens_1m")
echo ${dataset_list}
line_num_list=(7828 21189 30819)
cuda_num_list=(1 2 3)
echo ${line_num_list}
length=${#dataset_list[@]}
for ((i=0; i<${length}; i++));
do
{
    dataset=${dataset_list[i]}
    cuda_num=${cuda_num_list[i]}
    for model in din gru4rec sasrec bert4rec
    do
    {
        for type in _0_func_duet_fusion_vqvae_train # _0_func_base_train _0_func_duet_train _0_func_duet_vqvae_train _0_func_duet_fusion_train _0_func_duet_fusion_vqvae_train # _0_func_finetune_train
        do
        {
            bash ${type}.sh ${dataset} ${model} ${cuda_num}
        } &
        done
        # bash _0_func_base_train.sh ${dataset} ${model} 1 &
        # bash _0_func_duet_train.sh ${dataset} ${model} 2 &
        # bash _0_func_duet_fusion_train.sh ${dataset} ${model} 3 &
    } &
    done
    # bash _0_func_duet_fusion_vqvae_train.sh ${dataset} din 0 &
    # bash _0_func_duet_fusion_vqvae_train.sh ${dataset} gru4rec 1 &
    # bash _0_func_duet_fusion_vqvae_train.sh ${dataset} sasrec 2 &
    # bash _0_func_duet_fusion_vqvae_train.sh ${dataset} bert4rec 3 &
    
    # bash _0_func_apg_train.sh ${dataset} din 0 &
    # bash _0_func_apg_train.sh ${dataset} gru4rec 1 &
    # bash _0_func_apg_train.sh ${dataset} sasrec 2 &
    # bash _0_func_apg_train.sh ${dataset} bert4rec 3 &

    # bash _0_func_apg_fusion_train.sh ${dataset} din 3 &
    # bash _0_func_apg_fusion_train.sh ${dataset} gru4rec 2 &
    # bash _0_func_apg_fusion_train.sh ${dataset} sasrec 1 &
    # bash _0_func_apg_fusion_train.sh ${dataset} bert4rec 0 &
    
    # bash _0_func_duet_train.sh ${dataset} din 1 &
    # bash _0_func_duet_train.sh ${dataset} gru4rec 1 &
    # bash _0_func_duet_train.sh ${dataset} sasrec 0 &
    # bash _0_func_duet_train.sh ${dataset} bert4rec 0 &

    # bash _0_func_duet_train.sh ${dataset} sasrec 0 &
    # bash _0_func_duet_fusion_train.sh ${dataset} sasrec 0 &
    # bash _0_func_duet_fusion_vqvae_train.sh ${dataset} sasrec 0 &
} &
done
wait # 等待所有任务结束
date
# bash _0_func_base_train.sh amazon_beauty sasrec 0
# bash _0_func_finetune_train.sh amazon_beauty sasrec 0
# bash _0_func_duet_train.sh amazon_beauty sasrec 0