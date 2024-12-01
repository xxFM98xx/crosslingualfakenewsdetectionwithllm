from classes.InformationExtractor import InformationExtractor
import os
import pandas as pd
from utils import read_csv_files_from_directory, get_language_from_filename, save_extracted_information


if __name__ == "__main__":
    # Directory where the CSV files are located
    directory = r"C:\Users\FilmonMesfun\Desktop\MT_FM\cross-lingual-fake-news-detection-with-llm\Dataset\ProcessedDataset\"
    rationale_column_names = ['cot_linguistic_style_google','cot_common_sense_google','cot_linguistic_style_llm','cot_common_sense_llm','cot_linguistic_style_source','cot_common_sense_source']
    rationale_column_names_arg = ['cot_linguistic_style','cot_common_sense']
    # Read CSV files from the directory
    dataframes = read_csv_files_from_directory(directory)
        

    # Output the number of read DataFrames
    print(f"Amount of read DataFrames: {len(dataframes)}")
    
    # Create the 'extracted' subdirectory if it does not exist
    output_directory = os.path.join(directory, "extracted")
    os.makedirs(output_directory, exist_ok=True)
    
    # Create an InformationExtractor object and save the extracted information
    for filename, df in dataframes:
        try:
            
            language = get_language_from_filename(filename)
            print(f"Extracting information for file: {filename} with language: {language}")
            print(f"Columns: {df.columns}")
            
            # Check if all rationale_column_names are present in the DataFrame
            if all(column in df.columns for column in rationale_column_names):
                extractor = InformationExtractor(df=df, perspectives=['linguistic style', 'common sense'], rationale_column_names=rationale_column_names)
                extracted_df = extractor.extract_all_rationales_labels(language=language)
                save_extracted_information(extracted_df, output_directory, filename)
                print(f"Information extracted successfully for file: {filename}")
                
            # Check if all rationale_column_names_arg are present in the DataFrame
            else:
                print(f"Not all required columns found in file: {filename}.")
                extractor = InformationExtractor(df=df, perspectives=['linguistic style', 'common sense'], rationale_column_names=rationale_column_names_arg)
                extracted_df = extractor.extract_all_rationales_labels(language=language)
                save_extracted_information(extracted_df, output_directory, filename)
                print(f"Information extracted successfully for file: {filename}") 

        except KeyError as e:
            print(f"Error extracting information for file: {filename}. Missing column: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while extracting information for file: {filename}. Error: {e}")