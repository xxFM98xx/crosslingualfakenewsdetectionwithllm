


export SEED=3759
#ADJUST WHETHER TO USE GPU OR NOT.
export GPU=0 
export LR=5e-5
#ADJUST USED CLASSIFIER MODEL HERE. CHOOSE FROM:['RoBERTa_MLP', 'RoBERTa_CNN_MLP','ARG']
export MODEL_NAME="RoBERTa_CNN_MLP"

#DOES NOT NEED TO BE ADJUSTED
export LANGUAGE="en"

#PATH TO THE DATA SET TO BE USED FOR TRAINING ADJUST IF OTHER DATASET IS USED
export ROOT_PATH="cross-lingual-fake-news-detection-with-llm/Dataset/ProcessedDataset/prepped/falcon_7b_arg_chinese/source"

#MUST BE ADAPTED IF DIFFERENT SLM IS USED
export BERT_PATH="cross-lingual-fake-news-detection-with-llm/ARG/slm/xlm-roberta-base-local"

#SHOULD BE UNIQUE EXPERIMENT IN GENERAL THE SCHEMA THAT HAS BEEN USED: LLM_DATASET_DATASETVARIANT_CLASSIFIER
export DATA_NAME="falcon_7b_arg_chinese_source_RoBERTa_CNN_MLP"
export DATA_TYPE="rationale"
export RATIONALE_WEIGHT=1.5
export LLM_PREDICTOR_WEIGHT=1.0
export SAVE_LOG_DIR="./logs/$DATA_NAME"
export USE_RATIONALES=True
export TENSORBOARD_DIR="./logs/tensorlog/$DATA_NAME"

# Activate your virtual environment
source ../venv/bin/activate

# Run your Python script
python ../main.py     --seed $SEED     --gpu $GPU     --lr $LR     --model_name $MODEL_NAME     --language $LANGUAGE     --root_path $ROOT_PATH     --bert_path $BERT_PATH     --data_name $DATA_NAME     --data_type $DATA_TYPE     --rationale_usefulness_evaluator_weight $RATIONALE_WEIGHT     --llm_judgment_predictor_weight $LLM_PREDICTOR_WEIGHT     --save_log_dir $SAVE_LOG_DIR     --use_rationales $USE_RATIONALES     --tensorboard_dir $TENSORBOARD_DIR
