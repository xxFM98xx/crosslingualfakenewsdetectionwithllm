# Cross-language Fake News Detection using Large Language Models

This repository contains the code for the master thesis of Filmon Mesfun at the University of Ulm with the title **"Cross-language Fake News Detection using Large Language Models”**.

## Abstract

*The widespread distribution of fake news poses a serious societal threat, especially in an era dominated by AI-generated content that often outperforms human performance in terms of authenticity. Despite considerable progress in detecting fake news in English, research on low-resource languages is still limited. In this study, a translation-based approach to cross-lingual fake news detection is investigated using decoder-based large language models (LLMs) and encoder-based small language models (SLMs). Texts in Arabic, Bengali, Chinese and German are translated into English by an LLM to generate English rationales, hypothesizing that LLMs trained in English will show improved English reasoning in classification tasks. The generated English rationales are then fed to an SLM-based classifier with the texts in the source language. The work investigates two questions: the usability of LLMs for fake news recognition by pre-translating and generating English rationales and the impact of language distribution in LLM training data on recognition performance. With regard to the approach of pre-translating texts with LLM, the results were mixed. On the one hand, the observation of rationale generation performance showed variations depending on the perspective used. Furthermore, variations in performance also occurred for a single perspective, possibly due to the fact that the LLMs do not have task-specific or dataset-specific knowledge, as it is the case for a classifier trained on the dataset. When comparing the generation of rationales in English with those in the source language, there was an improvement in 40 of the 62 classifications. LLM classifiers with broader language support performed better. Looking at the trained classifiers augmented with rationales, the performance was more stable compared to LLM-only Performances, with a slight improvement in recognizing real messages than fake messages. This could be due to the imbalance of classes in the dataset. A comparison of the performance between datasets enriched with English and datasets enriched with the source language shows an improvement in 40 out of 90 cases. This improvement also depends on the extent to which the LLM used supports the language. However, the dedicated rational module of the ARG network did not show a significant advantage over other classifiers when it came to filtering out less useful rationales or extracting potential rationales. Furthermore, significant correlations were found between the influence of language distribution in the LLM training dataset and cross-linguistic performance. This suggests that language distribution has an important influence on LLM performance in cross-lingual scenarios.*

## Introduction

This project aims to detect fake news across different languages using large language models (LLMs). It includes the pre-processing of multilingual datasets, the generation of rationales using LLMs and the training of classifiers to detect fake news.

```
## Projektstructure
|-- main.py
|-- README.md
|-- PREPROCESSING_RATIONALE_GENERATION_LLM.md
|-- requirements.txt
|-- classes
  |-- CorrelationAnalysisLLM.py
  |-- Evaluator.py
  |-- InformationExtractor.py
  |-- ModelInputOutputHandlers.py
  |-- ModelsLoader.py
  |-- Pipeline.py
  |-- PreprocessingTranslation.py
  |-- PreTranslationsModule.py
  |-- RationaleModule.py
  |-- TrainingDataPrepperModule.py
|-- utils
  |-- data_prepper_script.py
  |-- evaluation_script.py
  |-- extraction_script.py
  |-- __init__.py
|-- scripts
  |-- run_falcon_7b_arg_chinese_gpu_slurm.sh
|-- model_training
  |-- grid_search.py
  |-- main.py
  |-- MODEL_TRAINING.md
  |-- requirements_models.txt
  |-- models
    |-- arg.py
    |-- argd.py
    |-- baseline.py
    |-- layers.py
  |-- scripts
    |-- run_falcon_7b_arg_chinese_source_RoBERTa_CNN_MLP_rationales_True.sh
  |-- slm
  |-- utils
    |-- dataloader.py
    |-- utils.py
|-- Dataset
  |-- InitialDataset
  |-- ProcessedDataset
```

## Steps
For preprocessing and rational generation, see PREPROCESSING_RATIONALE_GENERATION_LLM.md
For subsequent model training, see MODEL_TRAINING.md
