def read_csv_files_from_directory(directory_path):
    """ 
    Read all CSV files in a given directory and return a list of tuples, where each tuple contains the filename and the corresponding DataFrame.
    
    :param directory_path: Path to the directory where the CSV files are located.
    :return: List of tuples (filename, DataFrame).
    """
    # List to store the filenames and DataFrames
    dataframes = []
    
    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        # Check if the file is a CSV file
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)
            # Add the filename and DataFrame to the list
            dataframes.append((filename, df))
    
    return dataframes
def get_language_from_filename(filename):
    """
    returns the language of the file based on the filename
    """
    
    language_mapping_=
    {
        "asdn": "ar",
        "ban": "bn",
        "fang": "de",
        "arg_chinese": "zh",
        "arg": "en"
    }
    for key, value in language_mapping_.items():
        if key in filename:
            return value
    return None

def save_extracted_information(extracted_df, output_directory, filename):
    """
    Save the extracted information to a CSV file in the specified directory.
    
    :param extracted_df: The DataFrame with the extracted information.
    :param output_directory: The directory where the file should be saved.
    :param filename: The name of the original file to create the name of the output file.
    """
    # Create the file path for the output file
    output_file_path = os.path.join(output_directory, f"extracted_{filename}")
    
    # Save the extracted information to a CSV file
    extracted_df.to_csv(output_file_path, index=False)
    print(f"Extracted information saved in: {output_file_path}")
    
def read_files_into_list_of_dicts_json(directory):
    """
    Reads all JSON files in the specified directory and returns a list of dictionaries.
    
    Args:
        directory (str): The directory where the JSON files are located.
    
    Returns:
        list: A list of dictionaries containing the content of the JSON files.
    """
    list_of_dicts = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):  
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    content = json.load(f)
                    filename_without_extension = os.path.splitext(file)[0]
                    list_of_dicts.append({filename_without_extension: content})

    return list_of_dicts

def read_files_into_list_of_dicts_csv(directory):
    """
    Reads all CSV files in the specified directory and returns a list of dictionaries.
    
    Args:
        directory (str): The directory where the CSV files are located.
    
    Returns:
        list: A list of dictionaries containing the content of the CSV files.
    """
    
    list_of_dicts = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):  
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                filename_without_extension = os.path.splitext(file)[0]
                list_of_dicts.append({filename_without_extension: df})

    return list_of_dicts

def combine_json_files_in_subdir(subdir_path, output_file):
    """
    Combines all JSON files in the specified subdirectory into a single JSON file.
    
    Args:
        subdir_path (str): The path to the subdirectory containing the JSON files.
        output_file (str): The path to the output file where the combined data should be saved.
    """
    combined_data = []
    for root, dirs, files in os.walk(subdir_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    combined_data.extend(data)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=4)

def extract_ds_from_text(text:str):

    model_name = ['arg_chinese','arg','asdn','ban','fang']
    for model in model_name:
        if model in text.lower():
            return model
    
    return None

def extract_source_type_from_column(text:str):

    source_types = ['source','google','llm']
    for source_type in source_types:
        if source_type in text.lower():
            return source_type
    
    return None

def extract_classifer_from_column(text:str):
    classifier = ['arg','roberta_mlp','roberta_cnn_mlp']
    for classifer in classifier:
        if classifer in text.lower():
            return classifer
    return None

def extract_llm_from_column(text:str):
    llms = ['llama2','llama3.1','polylm','qwen2','seallm3','phoenix','gemma1.1','falcon']
    for llm in llms:
        if llm in text.lower():
            return llm
    return None

