# LSCDBenchmark

Lexical Semantic Change Detection (LSCD) is a field of NLP that studies methods automating the analysis of changes in word meanings over time. In recent years, this field has seen much development in terms of models, datasets and tasks [[1]](#1). This has made it hard to keep a good overview of the field. Additionally, with the multitude of possible options for preprocessing, data cleaning, dataset versions, model parameter choice or tuning, clustering algorithms, and change measures a shared testbed with common evaluation setup is needed in order to precisely reproduce experimental results. Hence, we present a benchmark repository implementing evaluation procedures for models on most available LSCD datasets.

## Documentation

[//]: # "[Latest Released Version](https://lscdbenchmark.readthedocs.io/en/stable/] \ "
[Main Branch Version (may include documentation fixes)](https://lscdbenchmark.readthedocs.io/en/latest/)

## Installation instructions

To get started, make sure that you have **Python `3.10.0`**. After that, clone the repository, then create a new virtual environment:

```sh
conda create -n lscdb python=3.10 pytorch=2.7.0 hydra-core=1.2.0 pydantic=1.10.2 tqdm=4.64.1 pandas=1.5.0 GitPython=3.1.31 gdown=5.2.0 pandera=0.12.0 matplotlib=3.6.0 transformers=4.54.1 sentence-transformers=5.0.0 more-itertools=8.14.0 pytest=7.3.1 -c pytorch -c conda-forge -y 
conda activate lscdb
pip install chinese-whispers==0.8.0
pip install git+https://github.com/nvanva/deepmistake@v3.0.0-alpha
```

## Getting Started

`LSCDBenchmark` heavily relies on [Hydra](https://hydra.cc/) for easily configuring experiments.

By running `python main.py`, the tool will guide you towards specifying some of its required parameters. The main parameters are:

- dataset
- evaluation
- task

From the shell, Hydra will ask you to provide values for all these parameters,
and will provide you with a list of options.
Once you select a value for each of these parameters, you might have to input
other, deeply nested required parameters. You can define a script to run your
experiments if you constantly find yourself typing the same command, as these
can get quite verbose.

An example, using the dataset `dwug_de`, with model `apd_compare_all` using BERT
as a WiC model and evaluating against graded change labels would be the
following:

```bash
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
```

Here, we chose `contextual_embedder` as a word-in-context model. This model
requires a `ckpt` parameter, which represents any model stored in [Huggingface
Hub](https://huggingface.co/models), like `bert-base-cased`,
`bert-base-uncased`, `xlm-roberta-large`, or
`dccuchile/bert-base-spanish-wwm-cased`.

`contextual_embedder` can also accept a `gpu` parameter. This parameter takes an
integer, and represents the ID of a certain GPU (there might be multiple on a
single machine).

## Running in inference mode

If you don't want to evaluate a model, you can use tilde notation (~) to remove a certain required parameter. For example, to run the previous command without any evaluation, you can run the following:

```bash
python main.py \
  dataset=dwug_de_210 \
  dataset/split=dev \
  dataset/spelling_normalization=german \
  dataset/preprocessing=normalization \
  task=lscd_graded \
  task/lscd_graded@task.model=apd_compare_all \
  task/wic@task.model.wic=contextual_embedder \
  task/wic/metric@task.model.wic.similarity_metric=cosine \
  task.model.wic.ckpt=bert-base-german-cased \
  ~evaluation
```

## Possible issues

- you may have to adjust the CUDA version specified in requirements.txt to your local requirements for the GPU.

## References

<a id="1">[1]</a>
Dominik Schlechtweg. 2023. [Human and Computational Measurement of Lexical Semantic Change](http://dx.doi.org/10.18419/opus-12833). PhD thesis. University of Stuttgart.
