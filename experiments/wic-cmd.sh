#!/bin/bash

echo "Starting experiments..."
echo "Evaluation with DeepMistake Model"
echo "---------------------------------"

#############################################################################################################
# # English - DeepMistake
# #############################################################################################################
# echo "Evaluation on English dataset (DeepMistake)..."
# for dataset in dwug_en_300 dwug_en_resampled_100; do
#     python main.py -m \
#         dataset=$dataset \
#         dataset/preprocessing=raw,normalization \
#         dataset/spelling_normalization=none,english \
#         dataset/split=full \
#         task=wic \
#         evaluation=wic \
#         evaluation/metric=spearman \
#         task/wic@task.model=deepmistake \
#         task/wic/dm_ckpt@task.model.ckpt=WIC_DWUG+XLWSD,WIC+RSS+DWUG+XLWSD \
#         +task/wic/metric@task.model.similarity_metric=cosine
# done

# #############################################################################################################
# # German - DeepMistake
# #############################################################################################################
# echo "Evaluation on German dataset (DeepMistake)..."
# for dataset in dwug_de_300 dwug_de_resampled_100 discowug_200; do
#     python main.py -m \
#         dataset=$dataset \
#         dataset/preprocessing=raw,normalization \
#         dataset/spelling_normalization=none,german \
#         dataset/split=full \
#         task=wic \
#         evaluation=wic \
#         evaluation/metric=spearman \
#         task/wic@task.model=deepmistake \
#         task/wic/dm_ckpt@task.model.ckpt=WIC_DWUG+XLWSD,WIC+RSS+DWUG+XLWSD \
#         +task/wic/metric@task.model.similarity_metric=cosine
# done

# #############################################################################################################
# # Swedish - DeepMistake
# #############################################################################################################
# echo "Evaluation on Swedish dataset (DeepMistake)..."
# for dataset in dwug_sv_300 dwug_sv_resampled_100; do
#     python main.py -m \
#         dataset=$dataset \
#         dataset/preprocessing=raw,normalization \
#         dataset/spelling_normalization=none,swedish \
#         dataset/split=full \
#         task=wic \
#         evaluation=wic \
#         evaluation/metric=spearman \
#         task/wic@task.model=deepmistake \
#         task/wic/dm_ckpt@task.model.ckpt=WIC_DWUG+XLWSD,WIC+RSS+DWUG+XLWSD \
#         +task/wic/metric@task.model.similarity_metric=cosine
# done

# #############################################################################################################
# # Russian, Norwegian, Chinese, Italian, Spanish - DeepMistake
# #############################################################################################################
# echo "Evaluation on Russian, Norwegian, Chinese, Italian, Spanish dataset (DeepMistake)..."
# for dataset in rusemshift_1_200 rusemshift_2_200 rushifteval1_200 rushifteval2_200 rushifteval3_200 \
#                nor_dia_change_1_101 nor_dia_change_2_101 dwug_it_300 chiwug_100 dwug_es_402; do
#     python main.py -m \
#         dataset=$dataset  \
#         dataset/preprocessing=raw,normalization \
#         dataset/spelling_normalization=none \
#         dataset/split=full \
#         task=wic \
#         evaluation=wic \
#         evaluation/metric=spearman \
#         task/wic@task.model=deepmistake \
#         task/wic/dm_ckpt@task.model.ckpt=WIC_DWUG+XLWSD,WIC+RSS+DWUG+XLWSD \
#         +task/wic/metric@task.model.similarity_metric=cosine
# done

#############################################################################################################
# XL-LeXeme
#############################################################################################################
echo "Evaluation on XL-LeXeme model..."

# German
for dataset in dwug_de_300 dwug_de_resampled_100 discowug_200; do
    python main.py -m \
        dataset=$dataset \
        dataset/preprocessing=raw,normalization \
        dataset/spelling_normalization=none,german \
        dataset/split=full \
        task=wic \
        evaluation=wic \
        evaluation/metric=spearman \
        task.model.ckpt=pierluigic/xl-lexeme \
        task/wic@task.model=contextual_embedder \
        task/wic/metric@task.model.similarity_metric=cosine
done

# English
for dataset in dwug_en_300 dwug_en_resampled_100; do
    python main.py -m \
        dataset=$dataset \
        dataset/preprocessing=raw,normalization \
        dataset/spelling_normalization=none,english \
        dataset/split=full \
        task=wic \
        evaluation=wic \
        evaluation/metric=spearman \
        task.model.ckpt=pierluigic/xl-lexeme \
        task/wic@task.model=contextual_embedder \
        task/wic/metric@task.model.similarity_metric=cosine
done

# Swedish
for dataset in dwug_sv_300 dwug_sv_resampled_100; do
    python main.py -m \
        dataset=$dataset \
        dataset/preprocessing=raw,normalization \
        dataset/spelling_normalization=none,swedish \
        dataset/split=full \
        task=wic \
        evaluation=wic \
        evaluation/metric=spearman \
        task.model.ckpt=pierluigic/xl-lexeme \
        task/wic@task.model=contextual_embedder \
        task/wic/metric@task.model.similarity_metric=cosine

done

# Other languages
for dataset in rusemshift_1_200 rusemshift_2_200 rushifteval1_200 rushifteval2_200 rushifteval3_200 \
               nor_dia_change_1_101 nor_dia_change_2_101 dwug_it_300 chiwug_100 dwug_es_402; do
    python main.py -m \
        dataset=$dataset \
        dataset/preprocessing=raw,normalization \
        dataset/spelling_normalization=none \
        dataset/split=full \
        task=wic \
        evaluation=wic \
        evaluation/metric=spearman \
        task.model.ckpt=pierluigic/xl-lexeme \
        task/wic@task.model=contextual_embedder \
        task/wic/metric@task.model.similarity_metric=cosine
done

# #############################################################################################################
# # XLM-R Large
# #############################################################################################################
# echo "Evaluation on XLM-R Large model..."

# # German
# for dataset in dwug_de_300 dwug_de_resampled_100 discowug_200; do
#     python main.py -m \
#         dataset=$dataset \
#         dataset/preprocessing=raw,normalization \
#         dataset/spelling_normalization=none,german \
#         dataset/split=full \
#         task=wic \
#         evaluation=wic \
#         evaluation/metric=spearman \
#         task.model.ckpt=xlm-roberta-large  \
#         task/wic@task.model=contextual_embedder \
#         task/wic/metric@task.model.similarity_metric=cosine
# done

# # English
# for dataset in dwug_en_300 dwug_en_resampled_100; do
#     python main.py -m \
#         dataset=$dataset \
#         dataset/preprocessing=raw,normalization \
#         dataset/spelling_normalization=none,english \
#         dataset/split=full \
#         task=wic \
#         evaluation=wic \
#         evaluation/metric=spearman \
#         task.model.ckpt=xlm-roberta-large  \
#         task/wic@task.model=contextual_embedder \
#         task/wic/metric@task.model.similarity_metric=cosine
# done

# # Swedish
# for dataset in dwug_sv_300 dwug_sv_resampled_100; do
#     python main.py -m \
#         dataset=$dataset \
#         dataset/preprocessing=raw,normalization \
#         dataset/spelling_normalization=none,swedish \
#         dataset/split=full \
#         task=wic \
#         evaluation=wic \
#         evaluation/metric=spearman \
#         task.model.ckpt=xlm-roberta-large  \
#         task/wic@task.model=contextual_embedder \
#         task/wic/metric@task.model.similarity_metric=cosine
# done

# # Other languages
# for dataset in rusemshift_1_200 rusemshift_2_200 rushifteval1_200 rushifteval2_200 rushifteval3_200 \
#                nor_dia_change_1_101 nor_dia_change_2_101 dwug_it_300 chiwug_100 dwug_es_402; do
#     python main.py -m \
#         dataset=$dataset \
#         dataset/preprocessing=raw,normalization \
#         dataset/spelling_normalization=none \
#         dataset/split=full \
#         task=wic \
#         evaluation=wic \
#         evaluation/metric=spearman \
#         task.model.ckpt=xlm-roberta-large  \
#         task/wic@task.model=contextual_embedder \
#         task/wic/metric@task.model.similarity_metric=cosine
# done

# echo "Done!"
