# Set environment variables
#ADAPT THE MODEL NAME HERE, WHEREBY THIS MUST MATCH A HUGGINGFACE REPOSITORY AS THE LLM IS LOADED FROM THERE.
#FOR THE EXPERIMENTAL SETUP THE LLMS [tiiuae/falcon-7b-instruct,FreedomIntelligence/phoenix-inst-chat-7b,DAMO-NLP-MT/polylm-chat-13b,
#Qwen/Qwen2-7B-Instruct,google/gemma-1.1-7b-it,meta-llama/Meta-Llama-3.1-8B-Instruct,meta-llama/Llama-2-7b-chat-hf,SeaLLMs/SeaLLMs-v3-7B-Chat] WERE USED 

export MODEL_NAME="tiiuae/falcon-7b-instruct"

export TEMPLATE_NAME="cot"
export QUANTIZED=False
export DISTRIBUTED=False
export NUM_WORKERS=1

#ADJUST THE COLUMN NAME HERE THAT CONTAINS THE TEXT TO BE PROCESSED.
export SOURCE_COLUMN="content"

#SET THE SOURCE LANGUAGE OF THE DATASET HERE.
export SOURCE_LANGUAGE="chinese"
export PERSPECTIVES="linguistic style,common sense"
export CLASS_BATCH_SIZE=4

#MUST BE SET TO FALSE IF RATIONALS ARE TO BE GENERATED FOR AN ENGLISH DATASET.
export RUN_PRETRANSLATE=True

#ADJUST JOB_NAME ACCORDINGLY. THE SCHEMA USED IS: run_LLM_DATASET
export JOB_NAME="run_falcon_7b_arg_chinese"

#THE PATH TO THE FILE TO BE PROCESSED MUST BE SET HERE.
export DATASET_FILENAME="cross-lingual-fake-news-detection-with-llm/Dataset/ProcessedDataset/arg_chinese_dataset_processed_translated.csv"

#THE PATH WHERE THE RESULT IS SAVED MUST BE SET HERE.
export SAVE_PATH_DIR="cross-lingual-fake-news-detection-with-llm/Dataset/ProcessedDataset/"


# Activate your virtual environment
source ../venv/bin/activate

# Run your Python script
python -u ../main.py "$@"


