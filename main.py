import os
from classes.ModelsLoader import ModelsLoader
from classes.Pipeline import Pipeline
import pandas as pd
from huggingface_hub import login

# Ensure the required environment variables are set, otherwise exit
MODEL_NAME = os.environ['MODEL_NAME']
TEMPLATE_NAME = os.environ['TEMPLATE_NAME']
QUANTIZED = os.environ['QUANTIZED'] == 'True'
DISTRIBUTED = os.environ['DISTRIBUTED'] == 'True'
NUM_WORKERS  = int(os.environ['NUM_WORKERS'])

SOURCE_COLUMN = os.environ['SOURCE_COLUMN']
SOURCE_LANGUAGE = os.environ['SOURCE_LANGUAGE']
PERSPECTIVES = os.environ['PERSPECTIVES'].split(',')
CLASS_BATCH_SIZE = int(os.environ['CLASS_BATCH_SIZE'])
RUN_PRETRANSLATE = os.environ['RUN_PRETRANSLATE'] == 'True'

# Get the job name and dataset filename for dynamic path construction
JOB_NAME = os.environ['JOB_NAME']
DATASET_FILENAME = os.environ['DATASET_FILENAME']

# SAVE PATH TO DIRECTORY FOR PROCESSED DATASETS
SAVE_PATH_DIR = os.environ['SAVE_PATH_DIR']

# Perform login if needed
#TODO TOKEN NEEDS TO BE SET
login(token="")

perspective_language_dict = {
    "common sense":{
        "arabic": "الحس السليم",
        "chinese": "常识",
        "english": "common sense",
        "bengali": "সাধারণ জ্ঞান",
        "german": "gesunder Menschenverstand"
    },
    "linguistic style":{
        "arabic":  "أسلوب لغوي",
        "chinese": "语言风格",
        "english": "linguistic style",
        "bengali": "ভাষাগত শৈলী",
        "german": "linguistischer Stil"
    }
}
perspectives_list = []
for perspective in PERSPECTIVES:
    if perspective in perspective_language_dict: 
        perspectives_list.append(perspective_language_dict[perspective])
    else:
        print(f"{perspective} ist nicht in PERSPECTIVES verfügbar.")
        raise ValueError(f"{perspective} ist nicht in PERSPECTIVES verfügbar.")


MAX_NEW_TOKENS_TRANSLATION = 500
        
def main():
    # Initialize the model loader
    models_loader = ModelsLoader(
        model_name=MODEL_NAME,
        quantized=QUANTIZED,
        distributed=DISTRIBUTED
    )
    # Set global seed for reproducibility
    models_loader.set_global_seed(42)

    # Setup distributed environment if applicable
    models_loader.setup_distributed()

    # Load model and tokenizer
    #dataset_path = f"/pfs/data5/home/ul/ul_student/ul_ler25/cross-lingual-fake-news-detection-with-llm/Dataset/ProcessedDataset/{DATASET_FILENAME}"
    dataset_path = DATASET_FILENAME
    dataset = pd.read_csv(dataset_path)
    tokenizer, model = models_loader.load_model()


    # Initialize the pipeline
    pipeline = Pipeline(
        dataset=dataset,
        source_column=SOURCE_COLUMN,
        model=model,
        tokenizer=tokenizer,
        source_language=SOURCE_LANGUAGE,
        perspectives=perspectives_list,
        class_batch_size=CLASS_BATCH_SIZE,
        model_name = MODEL_NAME,
        template_names = TEMPLATE_NAME,
        save_path_dir = SAVE_PATH_DIR,
        job_name = JOB_NAME,
        max_new_token_translate= MAX_NEW_TOKENS_TRANSLATION,
    )

    # Decide which function to run based on the environment variable
    if RUN_PRETRANSLATE:
        try:
            translated_llm_rationales_df, translated_google_rationales_df, source_rationales_df = pipeline.pretranslate_generate_rationale()
        except Exception as e:
            print(f"Generate rationales with pre-translation failed: {e}")
    else:
        try:
            source_rationales_only_df = pipeline.generate_rationales_without_translation()
        except Exception as e:
            print(f"Generate rationales without translation failed: {e}")

if __name__ == '__main__':
    try:
        main()
    except RuntimeError as e:
        if 'OutOfMemoryError' in str(e):
            print("OutOfMemoryError encountered even with batch size adjustments.")
        else:
            print(e)
            raise e
