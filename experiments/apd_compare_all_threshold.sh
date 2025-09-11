#!/bin/bash

run_experiment () {
  local dataset=$1
  local thresholds=$2
  local model=$3
  local eval=$4

  CUDA_VISIBLE_DEVICES=4 python main.py \
    dataset=$dataset \
    dataset/spelling_normalization=none \
    dataset/preprocessing=raw \
    dataset/split=comedi_test \
    task=lscd_graded \
    task/lscd_graded@task.model=apd_compare_all \
    task/wic@task.model.wic=contextual_embedder \
    task/wic/metric@task.model.wic.similarity_metric=cosine_cut_scaled \
    task.model.wic.similarity_metric.thresholds=$thresholds \
    task.model.wic.ckpt=$model \
    task.model.wic.gpu=1 \
    evaluation=$eval
}
 
# ------------------ CONFIG ------------------
models=("pierluigic/xl-lexeme" "sachinn1/xl-durel")

datasets_chinese=("chiwug_100")
datasets_english=("dwug_en_100" "dwug_en_201" "dwug_en_300" "dwug_en_resampled_100")
datasets_german=("dwug_de_110" "dwug_de_210" "dwug_de_230" "dwug_de_300" "dwug_de_resampled_100" "discowug_111" "discowug_200")
datasets_norwegian=("nor_dia_change_1_101" "nor_dia_change_2_101")
datasets_russian=("rusemshift_1_200" "rusemshift_2_200" "rushifteval1_200" "rushifteval2_200" "rushifteval3_200")
datasets_spanish=("dwug_es_400" "dwug_es_402")
datasets_swedish=("dwug_sv_100" "dwug_sv_201" "dwug_sv_300" "dwug_sv_resampled_100")
datasets_durel=("durel_300")
datasets_surel=("surel_300")

# ------------------ THRESHOLDS ------------------
get_thresholds () {
  local lang=$1
  local model=$2

  case "$lang-$model" in
    # pierluigic/xl-lexeme
    chinese-pierluigic/xl-lexeme)   echo "[0.4954352434934859,0.6498959346459109,0.6545358317876466]" ;;
    english-pierluigic/xl-lexeme)   echo "[0.41831215121205767,0.6067079642213467,0.6816738725189209]" ;;
    german-pierluigic/xl-lexeme)    echo "[0.33897473723297705,0.5652169831211743,0.6513646697218981]" ;;
    norwegian-pierluigic/xl-lexeme) echo "[0.3903189226117262,0.41404622984611356,0.5222997722408943]" ;;
    russian-pierluigic/xl-lexeme)   echo "[0.2494246766587687,0.5109239507222543,0.7485625465274197]" ;;
    spanish-pierluigic/xl-lexeme)   echo "[0.2124187364579282,0.45478048152290285,0.7884765488690593]" ;;
    swedish-pierluigic/xl-lexeme)   echo "[0.41889023140624687,0.6459711060295867,0.6719858070582518]" ;;

    # sachinn1/xl-durel
    chinese-sachinn1/xl-durel)   echo "[0.5768068139405162,0.6765002102091423,0.7927116093351737]" ;;
    english-sachinn1/xl-durel)   echo "[0.32535901204859474,0.48300452735758276,0.6115888591099456]" ;;
    german-sachinn1/xl-durel)    echo "[0.32997490400345,0.46455512615346295,0.5999736878501423]" ;;
    norwegian-sachinn1/xl-durel) echo "[0.2095633908339128,0.33858932660964186,0.4884544557715542]" ;;
    russian-sachinn1/xl-durel)   echo "[0.25539005328206443,0.49066763335561997,0.6147420218754426]" ;;
    spanish-sachinn1/xl-durel)   echo "[0.297084435862365,0.5208179680553657,0.62766399425355]" ;;
    swedish-sachinn1/xl-durel)   echo "[0.2898122908695373,0.45185373610991797,0.5643789604094149]" ;;

    # Durel/Surel placeholders
    durel-pierluigic/xl-lexeme) echo "[0.33897473723297705,0.5652169831211743,0.6513646697218981]" ;;
    surel-sachinn1/xl-durel) echo "[0.32997490400345,0.46455512615346295,0.5999736878501423]" ;;
  esac
}

# ------------------ RUN EXPERIMENTS ------------------
for model in "${models[@]}"; do
  for dataset in "${datasets_chinese[@]}"; do
    thresholds=$(get_thresholds "chinese" $model)
    run_experiment $dataset $thresholds $model change_graded
    run_experiment $dataset $thresholds $model compare
  done

  for dataset in "${datasets_english[@]}"; do
    thresholds=$(get_thresholds "english" $model)
    run_experiment $dataset $thresholds $model change_graded
    run_experiment $dataset $thresholds $model compare
  done

  for dataset in "${datasets_german[@]}"; do
    thresholds=$(get_thresholds "german" $model)
    run_experiment $dataset $thresholds $model change_graded
    run_experiment $dataset $thresholds $model compare
  done

  for dataset in "${datasets_norwegian[@]}"; do
    thresholds=$(get_thresholds "norwegian" $model)
    run_experiment $dataset $thresholds $model change_graded
    run_experiment $dataset $thresholds $model compare
  done

  for dataset in "${datasets_russian[@]}"; do
    thresholds=$(get_thresholds "russian" $model)
    run_experiment $dataset $thresholds $model compare
  done

  for dataset in "${datasets_spanish[@]}"; do
    thresholds=$(get_thresholds "spanish" $model)
    run_experiment $dataset $thresholds $model change_graded
    run_experiment $dataset $thresholds $model compare
  done

  for dataset in "${datasets_swedish[@]}"; do
    thresholds=$(get_thresholds "swedish" $model)
    run_experiment $dataset $thresholds $model change_graded
    run_experiment $dataset $thresholds $model compare
  done

  for dataset in "${datasets_durel[@]}"; do
    thresholds=$(get_thresholds "durel" $model)
    run_experiment $dataset $thresholds $model compare
  done

  for dataset in "${datasets_surel[@]}"; do
    thresholds=$(get_thresholds "surel" $model)
    run_experiment $dataset $thresholds $model compare
  done
done
