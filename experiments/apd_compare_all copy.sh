#!/bin/bash
## Unified experiment runner: multi-language, multi-model

run_experiment () {
  local dataset=$1
  local model=$2
  local eval=$3

  CUDA_VISIBLE_DEVICES=4 python main.py \
    dataset=$dataset \
    dataset/spelling_normalization=none \
    dataset/preprocessing=raw \
    dataset/split=comedi_test \
    task=lscd_graded \
    task/lscd_graded@task.model=apd_compare_all \
    task/wic@task.model.wic=contextual_embedder \
    task/wic/metric@task.model.wic.similarity_metric=cosine \
    task.model.wic.ckpt=$model \
    task.model.wic.gpu=1 \
    evaluation=$eval
}

# ------------------ CONFIG ------------------
models=("pierluigic/xl-lexeme" "sachinn1/xl-durel")

datasets_chinese=("chiwug_100")
datasets_english=("dwug_en_300" "dwug_en_resampled_100")
datasets_german=("dwug_de_300" "dwug_de_resampled_100" "discowug_200")
datasets_norwegian=("nor_dia_change_1_101" "nor_dia_change_2_101")
datasets_russian=("rusemshift_1_200" "rusemshift_2_200" "rushifteval1_200" "rushifteval2_200" "rushifteval3_200")
datasets_spanish=("dwug_es_402")
datasets_swedish=("dwug_sv_300" "dwug_sv_resampled_100")
datasets_durel=("durel_300")
datasets_surel=("surel_300")

# ------------------ RUN EXPERIMENTS ------------------
for model in "${models[@]}"; do
  for dataset in "${datasets_chinese[@]}"; do
    run_experiment $dataset $model change_graded
    run_experiment $dataset $model compare
  done

  for dataset in "${datasets_english[@]}"; do
    run_experiment $dataset $model change_graded
    run_experiment $dataset $model compare
  done

  for dataset in "${datasets_german[@]}"; do
    run_experiment $dataset $model change_graded
    run_experiment $dataset $model compare
  done

  for dataset in "${datasets_norwegian[@]}"; do
    run_experiment $dataset $model change_graded
    run_experiment $dataset $model compare
  done

  for dataset in "${datasets_russian[@]}"; do
    run_experiment $dataset $model compare
  done

  for dataset in "${datasets_spanish[@]}"; do
    run_experiment $dataset $model change_graded
    run_experiment $dataset $model compare
  done

  for dataset in "${datasets_swedish[@]}"; do
    run_experiment $dataset $model change_graded
    run_experiment $dataset $model compare
  done

  for dataset in "${datasets_durel[@]}"; do
    run_experiment $dataset $model compare
  done

  for dataset in "${datasets_surel[@]}"; do
    run_experiment $dataset $model compare
  done
done

# ------------------ EXTRA RUNS ------------------
CUDA_VISIBLE_DEVICES=0 python main.py \
  dataset=dwug_it_300 \
  dataset/spelling_normalization=none \
  dataset/preprocessing=raw \
  dataset/split=full \
  task=lscd_graded \
  task/lscd_graded@task.model=apd_compare_all \
  task/wic@task.model.wic=contextual_embedder \
  task/wic/metric@task.model.wic.similarity_metric=cosine \
  task.model.wic.ckpt=sachinn1/xl-durel2 \
  task.model.wic.gpu=1 \
  evaluation=change_graded

CUDA_VISIBLE_DEVICES=4 python main.py -m \
  dataset=dwug_de_300 \
  dataset/spelling_normalization=none \
  dataset/preprocessing=toklem,raw,tokenization,normalization,lemmatization \
  dataset/split=full \
  task=lscd_graded \
  task/lscd_graded@task.model=apd_compare_all \
  task/wic@task.model.wic=contextual_embedder \
  task/wic/metric@task.model.wic.similarity_metric=cosine \
  task.model.wic.ckpt=pierluigic/xl-lexeme,sachinn1/xl-durel2 \
  task.model.wic.gpu=1 \
  evaluation=change_graded



