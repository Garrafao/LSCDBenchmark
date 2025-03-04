: '
python main.py \
  dataset=dwug_de_210 \
  dataset/split=dev \
  dataset/spelling_normalization=german \
  dataset/preprocessing=raw \
  task=lscd_graded \
  task/lscd_graded@task.model=apd_compare_all \
  task/wic@task.model.wic=contextual_embedder \
  task/wic/metric@task.model.wic.similarity_metric=cosine \
  task.model.wic.ckpt=bert-base-german-cased \
  task.model.wic.gpu=0 \
  'dataset.test_on=[abbauen,abdecken,"abgebrüht"]' \
  evaluation=change_graded
'  
: '  
python main.py \
  dataset=dwug_de_210 \
  dataset/split=dev \
  dataset/spelling_normalization=german \
  dataset/preprocessing=toklem \
  task=lscd_graded \
  task/lscd_graded@task.model=apd_compare_all \
  task/wic@task.model.wic=contextual_embedder \
  task/wic/metric@task.model.wic.similarity_metric=cosine \
  task.model.wic.ckpt=bert-base-german-cased \
  task.model.wic.gpu=0 \
  evaluation=change_graded
'

python main.py \
  dataset=dwug_de_210 \
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

: '
python main.py \
  dataset=dwug_de_210 \
  dataset/split=dev \
  dataset/spelling_normalization=german \
  dataset/preprocessing=toklem \
  task=lscd_graded \
  task/lscd_graded@task.model=diasense_all \
  task/wic@task.model.wic=contextual_embedder \
  task/wic/metric@task.model.wic.similarity_metric=cosine \
  task.model.wic.ckpt=bert-base-german-cased \
  task.model.wic.gpu=0 \
  evaluation=change_graded
'
python main.py \
  dataset=dwug_de_210 \
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
