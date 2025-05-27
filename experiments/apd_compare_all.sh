#!/bin/bash
set -e

# echo "🚀 Running WIC model: deepmistake/WIC_DWUG+XLWSD"

# for model in apd_compare_all apd_compare_all_downsampled jsddot_all_downsampled diasense_all; do
#     echo "🔍 Running LSCD model: $model"
    
#     python main.py \
#         dataset=dwug_sv_300 \
#         dataset/spelling_normalization=swedish \
#         dataset/preprocessing=raw \
#         dataset/split=full \
#         task=lscd_graded \
#         task/lscd_graded@task.model=$model \
#         +task/wic/metric@task.model.wic.similarity_metric=cosine \
#         task/wic@task.model.wic=deepmistake \
#         task/wic/dm_ckpt@task.model.wic.ckpt=WIC+RSS+DWUG+XLWSD  \
#         +task.model.wic.gpu=1 \
#         evaluation=change_graded
    
#     echo "✅ Finished: $model"
#     echo "----------------------------------------"
# done


# echo "Running: xlm-roberta-large"
# for model in apd_compare_all apd_compare_all_downsampled jsddot_all_downsampled diasense_all; do
#   echo "🔍 Running LSCD model: $model"
#   python main.py \
#     dataset=dwug_sv_300 \
#     dataset/spelling_normalization=swedish \
#     dataset/preprocessing=raw \
#     dataset/split=full \
#     task=lscd_graded \
#     task/lscd_graded@task.model=$model \
#     task/wic@task.model.wic=contextual_embedder \
#     task/wic/metric@task.model.wic.similarity_metric=cosine \
#     task.model.wic.ckpt=xlm-roberta-large \
#     task.model.wic.gpu=1 \
#     evaluation=change_graded
# done
# echo "✅ Finished: $model"
# echo "----------------------------------------"

# echo "Running: xl-lexeme"
# for model in apd_compare_all apd_compare_all_downsampled jsddot_all_downsampled diasense_all; do
#   echo "🔍 Running LSCD model: $model"
#   python main.py \
#     dataset=dwug_sv_300 \
#     dataset/spelling_normalization=swedish \
#     dataset/preprocessing=raw \
#     dataset/split=full \
#     task=lscd_graded \
#     task/lscd_graded@task.model=$model \
#     task/wic@task.model.wic=contextual_embedder \
#     task/wic/metric@task.model.wic.similarity_metric=cosine \
#     task.model.wic.ckpt=pierluigic/xl-lexeme  \
#     task.model.wic.gpu=1 \
#     evaluation=change_graded
# done

# echo "✅ Finished: $model"
# echo "----------------------------------------"

# ## for resampled data 


# #!/bin/bash
# set -e

# echo "🚀 Running WIC model: deepmistake/WIC_DWUG+XLWSD"

# for model in apd_compare_all apd_compare_all_downsampled jsddot_all_downsampled diasense_all; do
#     echo "🔍 Running LSCD model: $model"
    
#     python main.py \
#         dataset=dwug_sv_300 \
#         dataset/spelling_normalization=swedish \
#         dataset/preprocessing=raw \
#         dataset/split=full \
#         task=lscd_graded \
#         task/lscd_graded@task.model=$model \
#         +task/wic/metric@task.model.wic.similarity_metric=cosine \
#         task/wic@task.model.wic=deepmistake \
#         task/wic/dm_ckpt@task.model.wic.ckpt=WIC+RSS+DWUG+XLWSD  \
#         +task.model.wic.gpu=1 \
#         evaluation=change_graded
    
#     echo "✅ Finished: $model"
#     echo "----------------------------------------"
# done


# echo "Running: xlm-roberta-large"
# for model in apd_compare_all apd_compare_all_downsampled jsddot_all_downsampled diasense_all; do
#   echo "🔍 Running LSCD model: $model"
#   python main.py \
#     dataset=dwug_sv_300 \
#     dataset/spelling_normalization=swedish \
#     dataset/preprocessing=raw \
#     dataset/split=full \
#     task=lscd_graded \
#     task/lscd_graded@task.model= apd_compare_all \
#     task/wic@task.model.wic=contextual_embedder \
#     task/wic/metric@task.model.wic.similarity_metric=cosine \
#     task.model.wic.ckpt=xlm-roberta-large \
#     task.model.wic.gpu=1 \
#     evaluation=change_graded
# done
# echo "✅ Finished: $model"
# echo "----------------------------------------"

# echo "Running: xl-lexeme"
# for model in apd_compare_all apd_compare_all_downsampled jsddot_all_downsampled diasense_all; do
#   echo "🔍 Running LSCD model: $model"
#   python main.py \
#     dataset=dwug_sv_300 \
#     dataset/spelling_normalization=swedish \
#     dataset/preprocessing=raw \
#     dataset/split=full \
#     task=lscd_graded \
#     task/lscd_graded@task.model=$model \
#     task/wic@task.model.wic=contextual_embedder \
#     task/wic/metric@task.model.wic.similarity_metric=cosine \
#     task.model.wic.ckpt=pierluigic/xl-lexeme  \
#     task.model.wic.gpu=1 \
#     evaluation=change_graded
# done

# echo "✅ Finished: $model"
# echo "----------------------------------------"

# ## for resampled data


# #!/bin/bash
# set -e

# echo "🚀 Running WIC model: deepmistake/WIC_DWUG+XLWSD"

# for model in apd_compare_all apd_compare_all_downsampled jsddot_all_downsampled diasense_all; do
#     echo "🔍 Running LSCD model: $model"
    
#     python main.py \
#         dataset=dwug_sv_300 \
#         dataset/spelling_normalization=swedish \
#         dataset/preprocessing=raw \
#         dataset/split=full \
#         task=lscd_graded \
#         task/lscd_graded@task.model=$model \
#         +task/wic/metric@task.model.wic.similarity_metric=cosine \
#         task/wic@task.model.wic=deepmistake \
#         task/wic/dm_ckpt@task.model.wic.ckpt=WIC+RSS+DWUG+XLWSD  \
#         +task.model.wic.gpu=1 \
#         evaluation=change_graded
    
#     echo "✅ Finished: $model"
#     echo "----------------------------------------"
# done


# echo "Running: xlm-roberta-large"
# for model in apd_compare_all apd_compare_all_downsampled jsddot_all_downsampled diasense_all; do
#   echo "🔍 Running LSCD model: $model"
#   python main.py \
#     dataset=dwug_sv_300 \
#     dataset/spelling_normalization=swedish \
#     dataset/preprocessing=raw \
#     dataset/split=full \
#     task=lscd_graded \
#     task/lscd_graded@task.model=$model \
#     task/wic@task.model.wic=contextual_embedder \
#     task/wic/metric@task.model.wic.similarity_metric=cosine \
#     task.model.wic.ckpt=xlm-roberta-large \
#     task.model.wic.gpu=1 \
#     evaluation=change_graded
# done
# echo "✅ Finished: $model"
# echo "----------------------------------------"

# echo "Running: xl-lexeme"
# for model in apd_compare_all apd_compare_all_downsampled jsddot_all_downsampled diasense_all; do
#   echo "🔍 Running LSCD model: $model"
#   python main.py \
#     dataset=dwug_sv_300 \
#     dataset/spelling_normalization=swedish \
#     dataset/preprocessing=raw \
#     dataset/split=full \
#     task=lscd_graded \
#     task/lscd_graded@task.model=$model \
#     task/wic@task.model.wic=contextual_embedder \
#     task/wic/metric@task.model.wic.similarity_metric=cosine \
#     task.model.wic.ckpt=pierluigic/xl-lexeme  \
#     task.model.wic.gpu=1 \
#     evaluation=change_graded
# done

# echo "✅ Finished: $model"
# echo "----------------------------------------"

# ## for resampled data 


# #!/bin/bash
# set -e

# echo "🚀 Running WIC model: deepmistake/WIC_DWUG+XLWSD"

# for model in apd_compare_all apd_compare_all_downsampled jsddot_all_downsampled diasense_all; do
#     echo "🔍 Running LSCD model: $model"

#     python main.py \
#         dataset=dwug_sv_resampled_100 \
#         dataset/spelling_normalization=swedish \
#         dataset/preprocessing=raw \
#         dataset/split=full \
#         task=lscd_graded \
#         task/lscd_graded@task.model=$model \
#         +task/wic/metric@task.model.wic.similarity_metric=cosine \
#         task/wic@task.model.wic=deepmistake \
#         task/wic/dm_ckpt@task.model.wic.ckpt=WIC+RSS+DWUG+XLWSD  \
#         +task.model.wic.gpu=1 \
#         evaluation=change_graded
    
#     echo "✅ Finished: $model"
#     echo "----------------------------------------"
# done


# echo "Running: xlm-roberta-large"
# for model in apd_compare_all apd_compare_all_downsampled jsddot_all_downsampled diasense_all; do
#   echo "🔍 Running LSCD model: $model"
#   python main.py \
#     dataset=dwug_sv_resampled_100  \
#     dataset/spelling_normalization=swedish \
#     dataset/preprocessing=raw \
#     dataset/split=full \
#     task=lscd_graded \
#     task/lscd_graded@task.model=$model \
#     task/wic@task.model.wic=contextual_embedder \
#     task/wic/metric@task.model.wic.similarity_metric=cosine \
#     task.model.wic.ckpt=xlm-roberta-large \
#     task.model.wic.gpu=1 \
#     evaluation=change_graded
# done
# echo "✅ Finished: $model"
# echo "----------------------------------------"

# echo "Running: xl-lexeme"
# for model in apd_compare_all apd_compare_all_downsampled jsddot_all_downsampled diasense_all; do
#   echo "🔍 Running LSCD model: $model"
#   python main.py \
#     dataset=dwug_sv_resampled_100  \
#     dataset/spelling_normalization=swedish \
#     dataset/preprocessing=raw \
#     dataset/split=full \
#     task=lscd_graded \
#     task/lscd_graded@task.model=$model \
#     task/wic@task.model.wic=contextual_embedder \
#     task/wic/metric@task.model.wic.similarity_metric=cosine \
#     task.model.wic.ckpt=pierluigic/xl-lexeme  \
#     task.model.wic.gpu=1 \
#     evaluation=change_graded
# done

# echo "✅ Finished: $model"
# echo "----------------------------------------"



########################################################################################################################################


## apd_comapre_all xl-lexeme

# echo "Running: xl-lexeme"
# for dataset in  chiwug_100 dwug_es_402 nor_dia_change_1_101 nor_dia_change_2_101 ; do
#   echo "🔍 Running LSCD model: $model"
#   python main.py \
#     dataset=$dataset  \
#     dataset/spelling_normalization=none \
#     dataset/preprocessing=raw \
#     dataset/split=full \
#     task=lscd_graded \
#     task/lscd_graded@task.model=apd_compare_all \
#     task/wic@task.model.wic=contextual_embedder \
#     task/wic/metric@task.model.wic.similarity_metric=cosine \
#     task.model.wic.ckpt=pierluigic/xl-lexeme  \
#     task.model.wic.gpu=1 \
#     evaluation=change_graded
# done

# echo "Running: xl-lexeme russian datasets"
# for dataset in rushifteval1_200 rushifteval2_200 rushifteval3_200 rusemshift_1_200 rusemshift_2_200 ; do
#   echo "🔍 Running dataset: $dataset"
#   python main.py \
#     dataset=$dataset  \
#     dataset/spelling_normalization=none \
#     dataset/preprocessing=raw \
#     dataset/split=full \
#     task=lscd_graded \
#     task/lscd_graded@task.model=apd_compare_all \
#     task/wic@task.model.wic=contextual_embedder \
#     task/wic/metric@task.model.wic.similarity_metric=cosine \
#     task.model.wic.ckpt=pierluigic/xl-lexeme  \
#     task.model.wic.gpu=1 \
#     evaluation=compare
# done


## apd comapre_all xlmr

# echo "Running: xlmr"
# for dataset in chiwug_100 dwug_es_402 nor_dia_change_1_101 nor_dia_change_2_101 ; do
#   echo "🔍 Running LSCD model: $dataset"
#   python main.py \
#     dataset=$dataset  \
#     dataset/spelling_normalization=none \
#     dataset/preprocessing=raw \
#     dataset/split=full \
#     task=lscd_graded \
#     task/lscd_graded@task.model=apd_compare_all \
#     task/wic@task.model.wic=contextual_embedder \
#     task/wic/metric@task.model.wic.similarity_metric=cosine \
#     task.model.wic.ckpt=xlm-roberta-large  \
#     task.model.wic.gpu=1 \
#     evaluation=change_graded
# done

# echo "Running: xlmr russian datasets"
# for dataset in rushifteval1_200 rushifteval2_200 rushifteval3_200 rusemshift_1_200 rusemshift_2_200 ; do
#   echo "🔍 Running dataset: $dataset"
#   python main.py \
#     dataset=$dataset  \
#     dataset/spelling_normalization=none \
#     dataset/preprocessing=raw \
#     dataset/split=full \
#     task=lscd_graded \
#     task/lscd_graded@task.model=apd_compare_all \
#     task/wic@task.model.wic=contextual_embedder \
#     task/wic/metric@task.model.wic.similarity_metric=cosine \
#     task.model.wic.ckpt=xlm-roberta-large  \
#     task.model.wic.gpu=1 \
#     evaluation=compare
# done

## deepmistake

# for dataset in nordiachange_1 nordiachange_2 ; do
#     echo "🔍 Running dataset: $dataset"
    
#     python main.py \
#         dataset=$dataset \
#         dataset/spelling_normalization=none \
#         dataset/preprocessing=raw \
#         dataset/split=full \
#         task=lscd_graded \
#         task/lscd_graded@task.model=apd_compare_all \
#         +task/wic/metric@task.model.wic.similarity_metric=cosine \
#         task/wic@task.model.wic=deepmistake \
#         task/wic/dm_ckpt@task.model.wic.ckpt=WIC+RSS+DWUG+XLWSD  \
#         +task.model.wic.gpu=1 \
#         evaluation=change_graded
#     echo "----------------------------------------"
# done

for dataset in nor_dia_change_2_101; do
    echo "🔍 Running dataset: $dataset"
    
    python main.py \
        dataset=$dataset \
        dataset/spelling_normalization=none \
        dataset/preprocessing=raw \
        dataset/split=full \
        task=lscd_graded \
        task/lscd_graded@task.model=apd_compare_all \
        +task/wic/metric@task.model.wic.similarity_metric=cosine \
        task/wic@task.model.wic=deepmistake \
        task/wic/dm_ckpt@task.model.wic.ckpt=WIC_DWUG+XLWSD  \
        +task.model.wic.gpu=1 \
        evaluation=change_graded
    echo "----------------------------------------"
done


for dataset in rushifteval1_200 rushifteval2_200 rushifteval3_200 rusemshift_1_200 rusemshift_2_200 ; do
    echo "🔍 Running dataset: $dataset"
    
    python main.py \
        dataset=$dataset \
        dataset/spelling_normalization=none \
        dataset/preprocessing=raw \
        dataset/split=full \
        task=lscd_graded \
        task/lscd_graded@task.model=apd_compare_all \
        +task/wic/metric@task.model.wic.similarity_metric=cosine \
        task/wic@task.model.wic=deepmistake \
        task/wic/dm_ckpt@task.model.wic.ckpt=WIC+RSS+DWUG+XLWSD  \
        +task.model.wic.gpu=1 \
        evaluation=compare
    echo "----------------------------------------"
done

for dataset in rushifteval1_200 rushifteval2_200 rushifteval3_200 rusemshift_1_200 rusemshift_2_200 ; do
    echo "🔍 Running dataset: $dataset"
    
    python main.py \
        dataset=$dataset \
        dataset/spelling_normalization=none \
        dataset/preprocessing=raw \
        dataset/split=full \
        task=lscd_graded \
        task/lscd_graded@task.model=apd_compare_all \
        +task/wic/metric@task.model.wic.similarity_metric=cosine \
        task/wic@task.model.wic=deepmistake \
        task/wic/dm_ckpt@task.model.wic.ckpt=WIC_DWUG+XLWSD  \
        +task.model.wic.gpu=1 \
        evaluation=compare
    echo "----------------------------------------"
done













