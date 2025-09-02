!/bin/bash
## change graded

for dataset in dwug_sv_200 dwug_sv_201 dwug_sv_300 dwug_sv_resampled_100 \
               dwug_de_100 dwug_210 dwug_de_230 dwug_de_300 dwug_de_resampled_100 \
               discowug_110 discowug_111 discowug_200 \
               dwug_en_100 dwug_en_201 dwug_en_300 dwug_en_resampled_100 \
               chiwug_100 \
               dwug_es_400 dwug_es_402 \
               nor_dia_change_1_101 nor_dia_change_2_101; do
  CUDA_VISIBLE_DEVICES=3 python main.py \
    dataset=$dataset \
    dataset/spelling_normalization=none \
    dataset/preprocessing=raw \
    dataset/split=comedi_test \
    task=lscd_graded \
    task/lscd_graded@task.model=apd_compare_all \
    task/wic@task.model.wic=contextual_embedder \
    task/wic/metric@task.model.wic.similarity_metric=cosine \
    task.model.wic.ckpt=xlm-roberta-large \
    task.model.wic.gpu=1 \
    evaluation=change_graded
done


# for dataset in dwug_sv_200 dwug_sv_201 dwug_sv_300 dwug_sv_resampled_100 \
#                dwug_de_100 dwug_210 dwug_de_230 dwug_de_300 dwug_de_resampled_100 \
#                discowug_110 discowug_111 discowug_200 \
#                dwug_en_100 dwug_en_201 dwug_en_300 dwug_en_resampled_100 \
#               chiwug_100 \
#                dwug_es_400 dwug_es_402 \
#                nor_dia_change_1_101 nor_dia_change_2_101; do

#   CUDA_VISIBLE_DEVICES=3 python main.py \
#     dataset=$dataset \
#     dataset/spelling_normalization=none \
#     dataset/preprocessing=raw \
#     dataset/split=comedi_test \
#     task=lscd_graded \
#     task/lscd_graded@task.model=apd_compare_all \
#     task/wic@task.model.wic=contextual_embedder \
#     task/wic/metric@task.model.wic.similarity_metric=cosine \
#     task.model.wic.ckpt=pierluigic/xl-lexeme  \
#     task.model.wic.gpu=1 \
#     evaluation=change_graded
# done

# for dataset in dwug_sv_200 dwug_sv_201 dwug_sv_300 dwug_sv_resampled_100 \
#                dwug_de_100 dwug_210 dwug_de_230 dwug_de_300 dwug_de_resampled_100 \
#                discowug_110 discowug_111 discowug_200 \
#                dwug_en_100 dwug_en_201 dwug_en_300 dwug_en_resampled_100 \
#                chiwug_100 \
#                dwug_es_400 dwug_es_402 \
#                nor_dia_change_1_101 nor_dia_change_2_101; do
               
#   CUDA_VISIBLE_DEVICES=3 python main.py \
#     dataset=$dataset \
#     dataset/spelling_normalization=none \
#     dataset/preprocessing=raw \
#     dataset/split=comedi_test \
#     task=lscd_graded \
#     task/lscd_graded@task.model=apd_compare_all \
#     task/wic@task.model.wic=contextual_embedder \
#     task/wic/metric@task.model.wic.similarity_metric=cosine \
#     task.model.wic.ckpt=sachinn1/xl-durel \
#     task.model.wic.gpu=1 \
#     evaluation=change_graded
# done

## compare


# for dataset in rushifteval1_200 rushifteval2_200 rushifteval3_200 rusemshift_1_200 rusemshift_2_200 durel_300 surel_300; do
#   CUDA_VISIBLE_DEVICES=3 python main.py \
#     dataset=$dataset \
#     dataset/spelling_normalization=none \
#     dataset/preprocessing=raw \
#     dataset/split=comedi_test \
#     task=lscd_graded \
#     task/lscd_graded@task.model=apd_compare_all \
#     task/wic@task.model.wic=contextual_embedder \
#     task/wic/metric@task.model.wic.similarity_metric=cosine \
#     task.model.wic.ckpt=pierluigic/xl-lexeme  \
#     task.model.wic.gpu=1 \
#     evaluation=compare
# done

# for dataset in rushifteval1_200 rushifteval2_200 rushifteval3_200 rusemshift_1_200 rusemshift_2_200 durel_300 surel_300; do
#   CUDA_VISIBLE_DEVICES=3 python main.py \
#     dataset=$dataset \
#     dataset/spelling_normalization=none \
#     dataset/preprocessing=raw \
#     dataset/split=comedi_test \
#     task=lscd_graded \
#     task/lscd_graded@task.model=apd_compare_all \
#     task/wic@task.model.wic=contextual_embedder \
#     task/wic/metric@task.model.wic.similarity_metric=cosine \
#     task.model.wic.ckpt=sachinn1/xl-durel \
#     task.model.wic.gpu=1 \
#     evaluation=compare
# done

# for dataset in rushifteval1_200 rushifteval2_200 rushifteval3_200 rusemshift_1_200 rusemshift_2_200 durel_300 surel_300; do
#   CUDA_VISIBLE_DEVICES=3 python main.py \
#     dataset=$dataset \
#     dataset/spelling_normalization=none \
#     dataset/preprocessing=raw \
#     dataset/split=comedi_test \
#     task=lscd_graded \
#     task/lscd_graded@task.model=apd_compare_all \
#     task/wic@task.model.wic=contextual_embedder \
#     task/wic/metric@task.model.wic.similarity_metric=cosine \
#     task.model.wic.ckpt=xlm-roberta-large \
#     task.model.wic.gpu=1 \
#     evaluation=compare
# done




for dataset in dwug_es_400 dwug_es_402; do
  CUDA_VISIBLE_DEVICES=1 python main.py \
    dataset=$dataset\
    dataset/spelling_normalization=none \
    dataset/preprocessing=raw \
    dataset/split=comedi_test \
    task=lscd_graded \
    task/lscd_graded@task.model=apd_compare_all \
    task/wic@task.model.wic=contextual_embedder \
    task/wic/metric@task.model.wic.similarity_metric=cosine \
    task.model.wic.ckpt=pierluigic/xl-lexeme \
    task.model.wic.gpu=1 \
    evaluation=change_graded
done

for dataset in dwug_es_400 dwug_es_402; do
  CUDA_VISIBLE_DEVICES=1 python main.py \
    dataset=$dataset\
    dataset/spelling_normalization=none \
    dataset/preprocessing=raw \
    dataset/split=comedi_test \
    task=lscd_graded \
    task/lscd_graded@task.model=apd_compare_all \
    task/wic@task.model.wic=contextual_embedder \
    task/wic/metric@task.model.wic.similarity_metric=cosine \
    task.model.wic.ckpt=sachinn1/xl-durel \
    task.model.wic.gpu=1 \
    evaluation=change_graded
done

for dataset in dwug_es_400 dwug_es_402; do
  CUDA_VISIBLE_DEVICES=1 python main.py \
    dataset=$dataset\
    dataset/spelling_normalization=none \
    dataset/preprocessing=raw \
    dataset/split=comedi_test \
    task=lscd_graded \
    task/lscd_graded@task.model=apd_compare_all \
    task/wic@task.model.wic=contextual_embedder \
    task/wic/metric@task.model.wic.similarity_metric=cosine \
    task.model.wic.ckpt=xlm-roberta-large \
    task.model.wic.gpu=1 \
    evaluation=change_graded
done
