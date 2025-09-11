#!/bin/bash
set -e

for dataset in dwug_sv_300 dwug_sv_resampled_100 ; do
  CUDA_VISIBLE_DEVICES=3 python main.py \
    dataset=$dataset \
    dataset/spelling_normalization=none \
    dataset/preprocessing=raw \
    dataset/split=full \
    task=lscd_graded \
    task/lscd_graded@task.model=apd_compare_all \
    task/wic@task.model.wic=contextual_embedder \
    task/wic/metric@task.model.wic.similarity_metric=cosine \
    task.model.wic.ckpt=xlm-roberta-large \
    task.model.wic.gpu=1 \
    evaluation=change_graded
done

for dataset in dwug_sv_300 dwug_sv_resampled_100; do
  CUDA_VISIBLE_DEVICES=3 python main.py \
    dataset=$dataset \
    dataset/spelling_normalization=none \
    dataset/preprocessing=raw \
    dataset/split=full \
    task=lscd_graded \
    task/lscd_graded@task.model=apd_compare_all \
    task/wic@task.model.wic=contextual_embedder \
    task/wic/metric@task.model.wic.similarity_metric=cosine \
    task.model.wic.ckpt=pierluigic/xl-lexeme  \
    task.model.wic.gpu=1 \
    evaluation=change_graded
done


for dataset in dwug_sv_300 dwug_sv_resampled_100; do
    CUDA_VISIBLE_DEVICES=3 python main.py \
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
        evaluation=change_graded
done


for dataset in dwug_sv_300 dwug_sv_resampled_100; do
    CUDA_VISIBLE_DEVICES=3 python main.py \
        dataset=$dataset \
        dataset/spelling_normalization=none \
        dataset/preprocessing=raw \
        dataset/split=full \
        task=lscd_graded \
        task/lscd_graded@task.model=apd_compare_all \
        +task/wic/metric@task.model.wic.similarity_metric=cosine \
        task/wic@task.model.wic=deepmistake \
        task/wic/dm_ckpt@task.model.wic.ckpt=WIC_DWUG+XLWSD \
        +task.model.wic.gpu=1 \
        evaluation=change_graded
done



for dataset in dwug_de_300 dwug_de_resampled_100 discowug_200; do
  CUDA_VISIBLE_DEVICES=3 python main.py \
    dataset=$dataset \
    dataset/spelling_normalization=none \
    dataset/preprocessing=raw \
    dataset/split=full \
    task=lscd_graded \
    task/lscd_graded@task.model=apd_compare_all \
    task/wic@task.model.wic=contextual_embedder \
    task/wic/metric@task.model.wic.similarity_metric=cosine \
    task.model.wic.ckpt=xlm-roberta-large \
    task.model.wic.gpu=1 \
    evaluation=change_graded
done


for dataset in dwug_de_300 dwug_de_resampled_100 discowug_200; do
  CUDA_VISIBLE_DEVICES=3 python main.py \
    dataset=$dataset \
    dataset/spelling_normalization=none \
    dataset/preprocessing=raw \
    dataset/split=full \
    task=lscd_graded \
    task/lscd_graded@task.model=apd_compare_all \
    task/wic@task.model.wic=contextual_embedder \
    task/wic/metric@task.model.wic.similarity_metric=cosine \
    task.model.wic.ckpt=pierluigic/xl-lexeme  \
    task.model.wic.gpu=1 \
    evaluation=change_graded
done

for dataset in dwug_de_300 dwug_de_resampled_100 discowug_200; do
    CUDA_VISIBLE_DEVICES=3 python main.py \
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
        evaluation=change_graded
done

for dataset in dwug_de_300 dwug_de_resampled_100 discowug_200; do
    CUDA_VISIBLE_DEVICES=3 python main.py \
        dataset=$dataset \
        dataset/spelling_normalization=none \
        dataset/preprocessing=raw \
        dataset/split=full \
        task=lscd_graded \
        task/lscd_graded@task.model=apd_compare_all \
        +task/wic/metric@task.model.wic.similarity_metric=cosine \
        task/wic@task.model.wic=deepmistake \
        task/wic/dm_ckpt@task.model.wic.ckpt=WIC_DWUG+XLWSD \
        +task.model.wic.gpu=1 \
        evaluation=change_graded
done


for dataset in dwug_en_300 dwug_en_resampled_100; do
    CUDA_VISIBLE_DEVICES=3 python main.py \
        dataset=$dataset \
        dataset/spelling_normalization=none \
        dataset/preprocessing=raw \
        dataset/split=full \
        task=lscd_graded \
        task/lscd_graded@task.model=apd_compare_all \
        +task/wic/metric@task.model.wic.similarity_metric=cosine \
        task/wic@task.model.wic=deepmistake \
        task/wic/dm_ckpt@task.model.wic.ckpt=WIC_DWUG+XLWSD \
        +task.model.wic.gpu=1 \
        evaluation=change_graded
done

for dataset in dwug_en_300 dwug_en_resampled_100; do
    CUDA_VISIBLE_DEVICES=3 python main.py \
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
        evaluation=change_graded
done



for dataset in dwug_en_300 dwug_en_resampled_100; do
  CUDA_VISIBLE_DEVICES=3 python main.py \
    dataset=$dataset \
    dataset/spelling_normalization=none \
    dataset/preprocessing=raw \
    dataset/split=full \
    task=lscd_graded \
    task/lscd_graded@task.model=apd_compare_all \
    task/wic@task.model.wic=contextual_embedder \
    task/wic/metric@task.model.wic.similarity_metric=cosine \
    task.model.wic.ckpt=xlm-roberta-large \
    task.model.wic.gpu=1 \
    evaluation=change_graded
done



for dataset in dwug_en_300 dwug_en_resampled_100; do
  CUDA_VISIBLE_DEVICES=3 python main.py \
    dataset=$dataset \
    dataset/spelling_normalization=none \
    dataset/preprocessing=raw \
    dataset/split=full \
    task=lscd_graded \
    task/lscd_graded@task.model=apd_compare_all \
    task/wic@task.model.wic=contextual_embedder \
    task/wic/metric@task.model.wic.similarity_metric=cosine \
    task.model.wic.ckpt=pierluigic/xl-lexeme  \
    task.model.wic.gpu=1 \
    evaluation=change_graded
done

for dataset in dwug_it_300 chiwug_100 dwug_es_402 nor_dia_change_1_101 nor_dia_change_2_101 ; do
    CUDA_VISIBLE_DEVICES=3 python main.py \
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
        evaluation=change_graded
done


for dataset in dwug_it_300 chiwug_100 dwug_es_402 nor_dia_change_1_101 nor_dia_change_2_101 ; do
    CUDA_VISIBLE_DEVICES=3 python main.py \
        dataset=$dataset \
        dataset/spelling_normalization=none \
        dataset/preprocessing=raw \
        dataset/split=full \
        task=lscd_graded \
        task/lscd_graded@task.model=apd_compare_all \
        +task/wic/metric@task.model.wic.similarity_metric=cosine \
        task/wic@task.model.wic=deepmistake \
        task/wic/dm_ckpt@task.model.wic.ckpt=WIC_DWUG+XLWSD \
        +task.model.wic.gpu=1 \
        evaluation=change_graded
done

for dataset in dwug_it_300 chiwug_100 dwug_es_402 nor_dia_change_1_101 nor_dia_change_2_101 ; do
  CUDA_VISIBLE_DEVICES=3 python main.py \
    dataset=$dataset \
    dataset/spelling_normalization=none \
    dataset/preprocessing=raw \
    dataset/split=full \
    task=lscd_graded \
    task/lscd_graded@task.model=apd_compare_all \
    task/wic@task.model.wic=contextual_embedder \
    task/wic/metric@task.model.wic.similarity_metric=cosine \
    task.model.wic.ckpt=xlm-roberta-large \
    task.model.wic.gpu=1 \
    evaluation=change_graded
done

for dataset in dwug_it_300 chiwug_100 dwug_es_402 nor_dia_change_1_101 nor_dia_change_2_101 ; do
  CUDA_VISIBLE_DEVICES=3 python main.py \
    dataset=$dataset \
    dataset/spelling_normalization=none \
    dataset/preprocessing=raw \
    dataset/split=full \
    task=lscd_graded \
    task/lscd_graded@task.model=apd_compare_all \
    task/wic@task.model.wic=contextual_embedder \
    task/wic/metric@task.model.wic.similarity_metric=cosine \
    task.model.wic.ckpt=pierluigic/xl-lexeme  \
    task.model.wic.gpu=1 \
    evaluation=change_graded
done



for dataset in rushifteval1_200 rushifteval2_200 rushifteval3_200 rusemshift_1_200 rusemshift_2_200 ; do
    CUDA_VISIBLE_DEVICES=3 python main.py \
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
done


for dataset in rushifteval1_200 rushifteval2_200 rushifteval3_200 rusemshift_1_200 rusemshift_2_200 ; do
    CUDA_VISIBLE_DEVICES=3 python main.py \
        dataset=$dataset \
        dataset/spelling_normalization=none \
        dataset/preprocessing=raw \
        dataset/split=full \
        task=lscd_graded \
        task/lscd_graded@task.model=apd_compare_all \
        +task/wic/metric@task.model.wic.similarity_metric=cosine \
        task/wic@task.model.wic=deepmistake \
        task/wic/dm_ckpt@task.model.wic.ckpt=WIC_DWUG+XLWSD \
        +task.model.wic.gpu=1 \
        evaluation=compare
done

for dataset in rushifteval1_200 rushifteval2_200 rushifteval3_200 rusemshift_1_200 rusemshift_2_200 ; do
  CUDA_VISIBLE_DEVICES=3 python main.py \
    dataset=$dataset \
    dataset/spelling_normalization=none \
    dataset/preprocessing=raw \
    dataset/split=full \
    task=lscd_graded \
    task/lscd_graded@task.model=apd_compare_all \
    task/wic@task.model.wic=contextual_embedder \
    task/wic/metric@task.model.wic.similarity_metric=cosine \
    task.model.wic.ckpt=xlm-roberta-large \
    task.model.wic.gpu=1 \
    evaluation=compare
done


for dataset in rushifteval1_200 rushifteval2_200 rushifteval3_200 rusemshift_1_200 rusemshift_2_200 ; do
  CUDA_VISIBLE_DEVICES=3 python main.py \
    dataset=$dataset \
    dataset/spelling_normalization=none \
    dataset/preprocessing=raw \
    dataset/split=full \
    task=lscd_graded \
    task/lscd_graded@task.model=apd_compare_all \
    task/wic@task.model.wic=contextual_embedder \
    task/wic/metric@task.model.wic.similarity_metric=cosine \
    task.model.wic.ckpt=pierluigic/xl-lexeme  \
    task.model.wic.gpu=1 \
    evaluation=compare
done

for dataset in durel_300 surel_300; do
    CUDA_VISIBLE_DEVICES=3 python main.py \
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
done

for dataset in durel_300 surel_300; do
    CUDA_VISIBLE_DEVICES=3 python main.py \
        dataset=$dataset \
        dataset/spelling_normalization=none \
        dataset/preprocessing=raw \
        dataset/split=full \
        task=lscd_graded \
        task/lscd_graded@task.model=apd_compare_all \
        +task/wic/metric@task.model.wic.similarity_metric=cosine \
        task/wic@task.model.wic=deepmistake \
        task/wic/dm_ckpt@task.model.wic.ckpt=WIC_DWUG+XLWSD \
        +task.model.wic.gpu=1 \
        evaluation=compare
done

for dataset in durel_300 surel_300; do
  CUDA_VISIBLE_DEVICES=3 python main.py \
    dataset=$dataset \
    dataset/spelling_normalization=none \
    dataset/preprocessing=raw \
    dataset/split=full \
    task=lscd_graded \
    task/lscd_graded@task.model=apd_compare_all \
    task/wic@task.model.wic=contextual_embedder \
    task/wic/metric@task.model.wic.similarity_metric=cosine \
    task.model.wic.ckpt=xlm-roberta-large \
    task.model.wic.gpu=1 \
    evaluation=compare
done

for dataset in durel_300 surel_300; do
  CUDA_VISIBLE_DEVICES=3 python main.py \
    dataset=durel_300 \
    dataset/spelling_normalization=none \
    dataset/preprocessing=raw \
    dataset/split=full \
    task=lscd_graded \
    task/lscd_graded@task.model=apd_compare_all \
    task/wic@task.model.wic=contextual_embedder \
    task/wic/metric@task.model.wic.similarity_metric=cosine \
    task.model.wic.ckpt=pierluigic/xl-lexeme  \
    task.model.wic.gpu=1 \
    evaluation=compare
done



