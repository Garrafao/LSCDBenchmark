

python main.py \
  dataset=dwug_de_300 \
  dataset/split=dev \
  dataset/spelling_normalization=german \
  dataset/preprocessing=toklem \
  task=lscd_graded \
  task/lscd_graded@task.model=apd_compare_sampled \
  task/wic@task.model.wic=contextual_embedder \
  task/wic/metric@task.model.wic.similarity_metric=cosine \
  task.model.wic.ckpt=bert-base-german-cased \
  task.model.wic.gpu=1 \
  evaluation=change_graded

python main.py \
  dataset=dwug_de_300 \
  dataset/split=dev \
  dataset/spelling_normalization=german \
  dataset/preprocessing=toklem \
  task=lscd_graded \
  task/lscd_graded@task.model=diasense_sampled \
  task/wic@task.model.wic=contextual_embedder \
  task/wic/metric@task.model.wic.similarity_metric=cosine \
  task.model.wic.ckpt=bert-base-german-cased \
  task.model.wic.gpu=1 \
  evaluation=change_graded
