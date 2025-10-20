#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 python main.py -m \
#   dataset=dwug_de_300 \
#   dataset/spelling_normalization=none \
#   dataset/preprocessing=toklem,raw,tokenization,normalization,lemmatization \
#   dataset/split=comedi_test \
#   task=lscd_graded \
#   task/lscd_graded@task.model=apd_compare_all \
#   task/wic@task.model.wic=contextual_embedder \
#   task/wic/metric@task.model.wic.similarity_metric=cosine \
#   task.model.wic.ckpt=pierluigic/xl-lexeme,sachinn1/xl-durel \
#   task.model.wic.gpu=1 \
#   evaluation=compare


CUDA_VISIBLE_DEVICES=0 python main.py \
  dataset=dwug_de_300 \
  dataset/spelling_normalization=none \
  dataset/preprocessing= toklem,raw,tokenization,normalization,lemmatization \
  dataset/split=comedi_test \
  task=lscd_graded \
  task/lscd_graded@task.model=apd_compare_all \
  +task/wic/metric@task.model.wic.similarity_metric=cosine \
  task/wic@task.model.wic=deepmistake \
  task/wic/dm_ckpt@task.model.wic.ckpt=MCL,MCLen \
  evaluation=change_graded