
---

### **README.md for Rationale Generation and Preprocessing**

```markdown
# Rational generation and pre-processing

This directory contains code and scripts to generate rationales using LLMs to compare source language rationales against English rationales (RQ1), to preprocess datasets and to analyze the influence of language distribution on model performance (RQ2) as part of the master thesis **"Cross-language Fake News Detection using Large Language Models ”**.

## Content

### Classes
- **PreprocessingTranslation.py**: Preprocesses Datasets and translates non-English datasets into English using the Google Translator API. To be executed with `preprocess_and_translate`.

- **PreTranslationsModule.py**: Class for pre-translating Datasets using a LLM. It should be called with `Pipeline.py`.

- RationaleModule.py**: Contains the class `RationaleGenerator` to generate rationales using an LLM. It should be called via `Pipeline.py`.

- Pipeline.py**: Orchestrates pre-translation and rationale generation using the perspectives “Linguistic Style” and “Common Sense”. Use `generate_rationales_without_translation` for English datasets and `pretranslate_generate_rationale` for non-English ones.

- **InformationExtractor.py**: Extrahiert Labels mittels Majority Threshold Voting, entfernt Labels aus den Rationales und extrahiert Übersetzungen aus den LLM-Antworten. Verwenden Sie `extract_all_translations` und `extract_all_rationales_labels`.
- InformationExtractor.py**: Extracts labels using Majority Threshold Voting, removes labels from rationales and extracts translations from LLM responses. Contains `extract_all_translations` and `extract_all_rationales_labels` but should be called via **extraction_script.py**

- **CorrelationAnalysisLLM.py**: Berechnet Korrelationen sprachenübergreifend sowie sprachspezifisch, zur Beantwortung der Forschungsfrage 2 (RQ2).
- CorrelationAnalysisLLM.py**: Calculates correlations across languages and language-specific, to answer research question 2 (RQ2).

### Utils
- **data_prepper_script.py**: Prepares data for model training by creating datasets from the extracted Rationales and Fake News datasets.

- **evaluation_script.py**: Evaluates generated rationales and LLM performance and provides functions for showing the performance of the classifier services.

- **extraction_script.py**: Extracts rationales and labels from the LLM outputs.

### Scripts
- **run_falcon_7b_arg_chinese_gpu_slurm.sh**: Generates rationals using the Falcon 7B LLM on the ARGChinese dataset.

## Instructions for generating rationales and pre-processing
### Prerequisites

- Python 3.10.5
- Installation of the required packages with `requirements.txt`.
- Access data for the Google Translator API.
- Hugging Face Token (set in `main.py`).

### Steps

#### 1. Pre-process and translate datasets

### 1. Download Datasets

Links:
- ARG Datasets(Englisch und Chinesisch) (needs to be requested): [Anfrageformular](https://forms.office.com/pages/responsepage.aspx?    id=DQSIkWdsW0yxEjajBLZtrQAAAAAAAAAAAAO__QiMr41UQlhTMUVHTzFLVEowWDhCODgwUjZZOTVOMi4u&route=shorturl)
- BanFakeNews: [Kaggle](https://www.kaggle.com/datasets/cryptexcode/banfakenews)
- FANG-COVID: [GitHub](https://github.com/justusmattern/fang-covid)
- AFND: [Kaggle](https://www.kaggle.com/datasets/murtadhayaseen/arabic-fake-news-dataset-afnd/data)

### 2. Save Datasets

The data sets must be stored individually in the folder `cross-lingual-fake-news-detection-with-llm\Dataset\InitialDataset`

### 3. Create the venv and install the packages in `requirements.txt`.

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r [requirements.txt](http://_vscodecontentref_/0)

### 4. Execute `PreprocessingTranslator.py`.

Execute `PreprocessingTranslator.py` to perform the preprocessing as described in the thesis. Furthermore, access data for the Google Translator API is required, which must be obtained to translate the non-English data records into English.

### 5. Rationale Generation

A huggingface token must be set in `main.py`. A `.sh` file can be found in `scripts`, which generates the rationales for the LLM Falcon for the Chinese data set. This must be used/created for each LLM from the list `[tiiuae/falcon-7b-instruct, FreedomIntelligence/phoenix-inst-chat-7b, DAMO-NLP-MT/polylm-chat-13b, Qwen/Qwen2-7B-Instruct, google/gemma-1.1-7b-it, meta-llama/Meta-Llama-3. 1-8B-Instruct, meta-llama/Llama-2-7b-chat-hf, SeaLLMs/SeaLLMs-v3-7B-Chat]` in combination for each dataset from the list `[ARGENGLISH, ARG-CHINESE, BanFakeNews, FANG-COVID, AFND]` to reproduce the experiment.

### 6. Extraktion der Rationales
To extract the rationales from the scripts, the `InformationExtractor` class can be used with the `extraction_script.py`, which extracts the rationales stored in `cross-lingual-fake-news-detection-with-llm\Dataset\ProcessedDataset\` (can be changed in the script) into the `extracted` subdirectory (is created automatically), so that a final label is determined and labels are extracted from the rationales.

### 6.5 Evaluation
The `Evaluation` class can be used to evaluate the rationales generated by the LLM. This contains various functions to evaluate the LLMs rationales across datasets in a DataFrame (`cross_ds_evaluation()`), per dataset (`_all_llm_evaluation_into_dfs`) and at the level of an individual LLM (`evaluate_llm()`).

#### 6.5.1 evaluate_llm()
To evaluate the rationales of an LLM on a dataset, `Evaluator.evaluate_llm()` can be called, whereby the DataFrame containing the rationales must be filtered to 0 and 1 in the `predicted` column before the function can be called. It saves `llm_report.json` for each dataset variant in a subdirectory in the specified directory `report_path`.

#### 6.5.2 _all_llm_evaluation_into_dfs()
To transfer the `llm_report.json` saved by `evaluate_llm` in subdirectories in JSON to DataFrames or tables, as done in the thesis, for each dataset, `Evaluator._all_llm_evaluation_into_dfs()` can be used. To do this, `report_dir` must be specified, which should contain all `llm_report.json` datasets.

### 7. Erstellen der Datensätze für nachfolgendes Training der Modelle
The `Data_Prepper` script can be used to create datasets from the previously generated and subsequently extracted DataFrames, which are stored in the `extracted` subdirectory. This takes the DataFrames stored in `extracted` and creates a new subdirectory `prepped` at the level of the subdirectory `extracted` and creates a subdirectory there for each LLM and dataset combination, e.g. `_falcon_7b_arg_chinese`. This subdirectory `falcon_7b_arg_chinese` contains sub-subdirectories for each data set variant `source`, `google`, `llm`, which then contain `train.json`, `test.json` and `val.json`.

### RQ2
To investigate the influence of the language distribution in the training data on the performance:

1. first evaluate the LLM_Performances by `Evaluator.evaluate_llm()`, if not done before, and store them in `cross-lingual-fake-news-detection-with-llm\Reports`.
2. call `evaluation_script.py` by calling the functions `all_arg_llm_cross_ds_(LLM_REPORT_PATH)` so that a cross-dataset LLM Performances DataFrame is created, as done in the thesis.
3. call `evaluation_script.py` `transform_llm_into_la_info_supported(pd.read_csv(LLM_CROSS_DS_PATH))` to create the DataFrames `data_df`, `binary_data_df`, which contain the LLM Performances and the language distributions in the form of supported languages and numerical distributions respectively.
4. execute `CorrelationAnalysisLLM.py` to obtain correlations saved in DataFrames.

```
