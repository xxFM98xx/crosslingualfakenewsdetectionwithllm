import os
import pandas as pd
from classes.TrainingDataPrepperModule import DataPrepper
import json
from utils import read_csv_files_from_directory

# Base directory of the files
input_directory = r"C:\Users\FilmonMesfun\Desktop\MT_FM\cross-lingual-fake-news-detection-with-llm\Dataset\ProcessedDataset\extracted"
output_directory = r"C:\Users\FilmonMesfun\Desktop\MT_FM\cross-lingual-fake-news-detection-with-llm\Dataset\ProcessedDataset\prepped"

os.makedirs(output_directory, exist_ok=True)



# Column names for the different rationale sources
rationale_column_names_google = ['cot_linguistic_style_google', 'cot_common_sense_google']
rationale_column_names_llm = ['cot_linguistic_style_llm', 'cot_common_sense_llm']
rationale_column_names_source = ['cot_linguistic_style_source', 'cot_common_sense_source']
rationale_column_names =  ['cot_linguistic_style', 'cot_common_sense']


# Dictionary with the different perspectives in different languages
perspective_language_dict_list = [
    {
        "arabic": "الحس السليم",
        "chinese": "常识",
        "english": "common sense",
        "bengali": "সাধারণ জ্ঞান",
        "german": "gesunder Menschenverstand"
    },
    {
        "arabic":  "أسلوب لغوي",
        "chinese": "语言风格",
        "english": "linguistic style",
        "bengali": "ভাষাগত শৈলী",
        "german": "linguistischer Stil"
    }
]

# Dictionary with the different configurations for the rationale columns
rationale_configs = {
    "google": rationale_column_names_google,
    "llm": rationale_column_names_llm,
    "source": rationale_column_names_source,
}

# Mapping of the source column names containing the fake news text to the dataset names
source_column_dataset_mapping = {
    "arg_chinese": "content",
    "arg": "content",
    "asnd": "content",
    "ban": "content",
    "fang": "article"
}

# Retrieve all CSV files from the input directory
dataframes = read_csv_files_from_directory(input_directory)

# Output the number of read DataFrames
print(f"Amount of read DataFrames: {len(dataframes)}")
# Create an DataPrepper object and save 
for filename, df in dataframes:
    try:
        # DataFrame loaded from the CSV file in the current iteration of the loop iterating over the dataframes in the input directory
        extracted_df = pd.read_csv(csv_path)
        
        # Create a subdirectory for the current file by extracting the basename of the file(dataset name) without the extension
        file_basename = os.path.splitext(csv_file)[0]  # Dataframe name without the extension
        for key, value in source_column_dataset_mapping.items():
        # Check if the current key is contained in the filename
            if key in file_basename:
                # If found, the value is assigned as source_column
                source_column = value
                break
        
        subdir = os.path.join(output_directory, file_basename)
        os.makedirs(subdir, exist_ok=True)

        # Every configuration(source, llm, google) has its own subsubdirectory and the JSONs are saved there
        for subsubdir_name, rationale_columns in rationale_configs.items():
            # Create subsubdirectory
            subsubdir = os.path.join(subdir, subsubdir_name)
            os.makedirs(subsubdir, exist_ok=True)

            # DataPrepper for the respective rationale column
            try:
                if 'source' in subsubdir_name:
                    dataprepper = DataPrepper(df=extracted_df, rationale_column_names=rationale_columns, 
                                            perspectives=perspective_language_dict_list, 
                                            source_column=source_column)
                elif 'llm' in subsubdir_name:
                    dataprepper = DataPrepper(df=extracted_df, rationale_column_names=rationale_columns, 
                                                perspectives=perspective_language_dict_list, 
                                                source_column=f"llm_translated_{source_column}_extracted_translation")
                elif 'google' in subsubdir_name:
                    dataprepper = DataPrepper(df=extracted_df, rationale_column_names=rationale_columns, 
                                                perspectives=perspective_language_dict_list,
                                                source_column=f"translated_{source_column}")
                    
            
                # Retrieve the datasets train, val, test from the DataPrepper object for the respektive source type(source, google, llm)
                train, val, test = dataprepper.get_arg_datasets()
            except Exception as e:
                raise ValueError(f"Error processing the file {csv_file}: {e}")

            
            # Save the datasets as JSON files in the respective subsubdirectory
            with open(os.path.join(subsubdir, 'train.json'), 'w', encoding='utf-8') as train_file:
                json.dump(train, train_file, ensure_ascii=False, indent=4)
            
            with open(os.path.join(subsubdir, 'val.json'), 'w', encoding='utf-8') as val_file:
                json.dump(val, val_file, ensure_ascii=False, indent=4)
            
            with open(os.path.join(subsubdir, 'test.json'), 'w', encoding='utf-8') as test_file:
                json.dump(test, test_file, ensure_ascii=False, indent=4)
            
            print(f"Datasets for {csv_file} in {subsubdir_name} successfully processed and saved.")