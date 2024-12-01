import pandas as pd
from typing import List, Union
import os
import json
from google.cloud import translate_v2 as translate
from google.oauth2 import service_account
from transformers import AutoTokenizer
import numpy as np
import logging

class PreprocessingTranslator:
    """
    A class to handle loading, filtering, and translating DataFrame content using the Google Translate API.
    """

    def __init__(self, tokenizer_name: str, src_lang: str, credentials_path: str=None, tgt_lang: str = 'en', max_length: int = 1024, max_chars: int = 999999999999, log_file: str = 'preprocessing_log.txt'):
        """
        Initializes the DataFrameTranslator with a tokenizer and sets up logging.

        :param credentials_path: Path to the JSON file with Google Cloud credentials.
        :param tokenizer_name: Name of the tokenizer to use for length checking.
        :param src_lang: Source language for translation.
        :param tgt_lang: Target language for translation.
        :param max_length: Maximum length of the text to be translated.
        :param max_chars: Maximum number of characters to be translated to control costs.
        :param log_file: Path to the log file for preprocessing steps.
        """
        if credentials_path is not None:
            self.credentials = service_account.Credentials.from_service_account_file(credentials_path)
            self.translator = translate.Client(credentials=self.credentials)
        else:
            self.credentials = None
            self.translator = None
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_length = max_length
        self.max_chars = max_chars
        self.translated_chars = 0
        self.setup_logging(log_file)

    def setup_logging(self, log_file: str) -> None:
        """
        Sets up the logging configuration.

        :param log_file: Path to the log file.
        """
        logging.basicConfig(
            filename=log_file,
            filemode='w',
            format='%(asctime)s - %(message)s',
            level=logging.INFO
        )
        self.logger = logging.getLogger()
    
    def load_dataframe_from_json(self, file_paths: Union[str, List[str]]) -> None:
        """
        Loads one or more JSON files into a DataFrame.

        :param file_paths: A single file path or a list of file paths.
        """
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        dfs = [pd.read_json(file_path) for file_path in file_paths]
        self.df = pd.concat(dfs, ignore_index=True)

    def load_dataframe_from_csv(self, file_paths: Union[str, List[str]]) -> None:
        """
        Loads one or more CSV files into a DataFrame.

        :param file_paths: A single file path or a list of file paths.
        """
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        dfs = [pd.read_csv(file_path) for file_path in file_paths]
        self.df = pd.concat(dfs, ignore_index=True)

    def load_json_files(self, folder_path: str) -> None:
        """
        Loads multiple numbered JSON files from a folder into a DataFrame.

        :param folder_path: Path to the folder containing JSON files.
        """
        json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
        data = []

        for json_file in json_files:
            file_path = os.path.join(folder_path, json_file)
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                data.append(json_data)

        self.df = pd.DataFrame(data)

    def preprocess_dataframe(self, column_name: str, target_count: int) -> None:
        """
        Preprocesses the DataFrame by removing duplicates, empty values, rows with text lengths outside the maximum length
        and the interquartile range (IQR), and then randomly drops rows to achieve a target count.

        :param column_name: Name of the column to check text lengths.
        :param target_count: The desired number of remaining rows after preprocessing.
        """
        initial_count = len(self.df)
        self.logger.info(f"Initial number of articles: {initial_count}")

        # Remove duplicates
        self.df = self.df.drop_duplicates(subset=[column_name])
        duplicates_removed_count = initial_count - len(self.df)
        self.logger.info(f"Number of duplicate articles removed: {duplicates_removed_count}")

        # Remove empty or missing values
        self.df = self.df.dropna(subset=[column_name])
        self.df = self.df[self.df[column_name].str.strip() != '']
        missing_values_removed_count = initial_count - len(self.df) - duplicates_removed_count
        self.logger.info(f"Number of articles with missing or empty values removed: {missing_values_removed_count}")

        # Update initial count after removing duplicates and empty values
        initial_count = len(self.df)

        # Filter by maximum length
        tokenizer = self.tokenizer
        lengths = self.df[column_name].apply(lambda x: len(tokenizer(x)["input_ids"]))
        filtered_df_max_length = self.df[lengths <= self.max_length]
        max_length_filtered_count = initial_count - len(filtered_df_max_length)
        self.logger.info(f"Number of articles removed due to length > {self.max_length} tokens: {max_length_filtered_count}")

        # Update dataframe and lengths after max length filtering
        self.df = filtered_df_max_length
        lengths = self.df[column_name].apply(lambda x: len(tokenizer(x)["input_ids"]))  # Recalculate lengths

        # Filter by IQR
        q1 = np.percentile(lengths, 25)
        q3 = np.percentile(lengths, 75)
        iqr = q3 - q1
        lower_bound = max(0, q1 - 1.5 * iqr)  # Ensure lower bound is not negative
        upper_bound = q3 + 1.5 * iqr
        filtered_df_iqr = self.df[(lengths >= lower_bound) & (lengths <= upper_bound)]
        iqr_filtered_count = len(self.df) - len(filtered_df_iqr)
        self.logger.info(f"Number of articles removed due to being outside IQR [{lower_bound}, {upper_bound}]: {iqr_filtered_count}")

        # Update dataframe after IQR filtering
        self.df = filtered_df_iqr
        final_count = len(self.df)
        total_removed_count = initial_count - final_count
        self.logger.info(f"Total number of articles removed during preprocessing: {total_removed_count}")
        self.logger.info(f"Number of articles remaining after preprocessing: {final_count}")

        # Randomly drop rows to achieve target count
        if target_count < final_count:
            self.df = self.df.sample(n=target_count, random_state=42)
            self.logger.info(f"Number of articles remaining after random drop to achieve target count of {target_count}: {len(self.df)}")

    def translate_row(self, row: pd.Series, column_name: str) -> str:
        """
        Translates the content of a specific column of a DataFrame row using Google Translate API.

        :param row: A row from the DataFrame.
        :param column_name: The name of the column to be translated.
        :return: The translated text.
        """
        if 'translated_' + column_name in row:
            if row['translated_' + column_name] != None:
                self.logger.info("Translation already exists. Skipping translation.")
                return row['translated_' + column_name]

        article = row[column_name]
        print("article: "+article)
        if self.translated_chars + len(article) > self.max_chars:
            # Log the information and skip translation if character limit is reached
            self.logger.info("Translation limit reached. Stopping translation to avoid exceeding character limit.")
            return None

        translated_article = self.translator.translate(article, source_language=self.src_lang, target_language=self.tgt_lang)['translatedText']
        print("translated article: "+translated_article)
        self.logger.info("Translation successful.")
        self.translated_chars += len(article)
        return translated_article

    def translate_dataframe(self, source_column: str, target_column: str) -> None:
        """
        Translates the entire DataFrame and adds the translations as a new column.

        :param source_column: Name of the source column.
        :param target_column: Name of the target column for translations.
        """
        self.df[target_column] = self.df.apply(lambda row: self.translate_row(row, source_column), axis=1)
        # Drop rows where translation is None due to character limit reached
        #self.df = self.df.dropna(subset=[target_column])

    def count_long_articles(self, column_name: str, tokenizer_name: str) -> int:
        """
        Counts the number of articles that are longer than the maximum length.

        :param column_name: Name of the column to check.
        :param tokenizer_name: Name of the tokenizer to use.
        :return: Number of long articles.
        """
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        count = 0
        for article in self.df[column_name]:
            encoding = tokenizer(article, truncation=False)
            if len(encoding["input_ids"]) > self.max_length:
                count += 1
        return count

    def write_dataframe_to_csv(self, output_path: str) -> None:
        """
        Saves the translated DataFrame as a CSV file.

        :param output_path: Path to the output file.
        """
        self.df.to_csv(output_path, index=False)

    def count_total_characters(self, column_name: str) -> int:
        """
        Counts the total number of characters in a specified column of the DataFrame.

        :param column_name: Name of the column to count characters.
        :return: Total number of characters.
        """
        total_chars = self.df[column_name].apply(len).sum()
        return total_chars
    
    def preprocess_and_translate(self, column_name: str, target_count: int, source_column: str, target_column: str, save_path:str=None) -> None:
        """
        Preprocesses the DataFrame, drops rows to achieve a target count and translates the content of a specific column.
        
        :param column_name: Name of the column to check text lengths.
        :param target_count: The desired number of remaining rows after preprocessing.
        :param source_column: Name of the source column.
        :param target_column: Name of the target column for translations.
        :param save_path: Path to save the translated DataFrame. If None, the DataFrame is not saved.
        """
        print(self.df.head())
        print("Dataframe columns: "+self.df.columns)
        print(f"Number of articles in the DataFrame before preprocessing: {len(self.df)}")
        self.preprocess_dataframe(column_name, target_count)
        print(f"Number of articles in the DataFrame after preprocessing: {len(self.df)}")
        self.translate_dataframe(source_column, target_column)
        if save_path is not None:
            self.write_dataframe_to_csv(save_path)
            print(f"Translated DataFrame and saved to {save_path}")

        


if __name__ == "__main__":
    SYSTEM_PATH_APPENDIX = "cross-lingual-fake-news-detection-with-llm"
    CREDENTIALS_PATH = SYSTEM_PATH_APPENDIX + "/classes/cred2.json"
    

    # BanFakeNews dataset Path
    file_paths_ban_fake_news = [
        SYSTEM_PATH_APPENDIX + "/Dataset/BanFakeNews/Authentic-48K.csv",
        SYSTEM_PATH_APPENDIX + "/Dataset/BanFakeNews/Fake-1K.csv"
    ]
    BAN_FAKE_NEWS_SAVE_PATH = SYSTEM_PATH_APPENDIX + "/Dataset/ProcessedDataset/ban_fake_news_processed_translated.csv"

    # FANG-COVID dataset Path
    FOLDER_PATH_FANG_COVID = SYSTEM_PATH_APPENDIX + "/Dataset/InitialDataset/FANG-COVID/fang-covid/articles"
    FANG_COVID_SAVE_PATH = SYSTEM_PATH_APPENDIX + "/Dataset/ProcessedDataset/fang_covid_processed_translated.csv"


    #ARG Path
    file_paths_arg_dataset = [
        SYSTEM_PATH_APPENDIX + "/Dataset/ARG-Dataset/en-20240507T104445Z-001/en/train.json",
        SYSTEM_PATH_APPENDIX + "/Dataset/ARG-Dataset/en-20240507T104445Z-001/en/test.json",
        SYSTEM_PATH_APPENDIX + "/Dataset/ARG-Dataset/en-20240507T104445Z-001/en/val.json"
    ]

    ARG_SAVE_PATH = SYSTEM_PATH_APPENDIX + "/Dataset/ProcessedDataset/arg_dataset_processed.csv"
    
    #ARG Chinese
    file_paths_arg_dataset_chinese = [
        SYSTEM_PATH_APPENDIX + "/Dataset/InitialDataset/ARG-Dataset/zh-20240507T104447Z-001/zh/train.json",
        SYSTEM_PATH_APPENDIX + "/Dataset/InitialDataset/ARG-Dataset/zh-20240507T104447Z-001/zh/test.json",
        SYSTEM_PATH_APPENDIX + "/Dataset/InitialDataset/ARG-Dataset/zh-20240507T104447Z-001/zh/val.json"
    ]
    
    ARG_SAVE_PATH_CHINESE = SYSTEM_PATH_APPENDIX + "/Dataset/ProcessedDataset/arg_chinese_dataset_processed_translated.csv"

    # AFND Path
    AFND_PATH = SYSTEM_PATH_APPENDIX + "/Dataset/InitialDataset/AFND/AFND_full.csv"
    
    AFND_SAVE_PATH = SYSTEM_PATH_APPENDIX + "/Dataset/ProcessedDataset/afnd_dataset_processed_translated.csv"
    
    afnd_translator = PreprocessingTranslator(
        credentials_path=CREDENTIALS_PATH,
        src_lang="ar",
        tokenizer_name="DAMO-NLP-MT/polylm-13b",
        log_file="preprocessing_log_afnd.txt"
    )
    
    arg_network_chinese_translator = PreprocessingTranslator(
        credentials_path=CREDENTIALS_PATH,
        src_lang="zh",
        tokenizer_name="DAMO-NLP-MT/polylm-13b",
        log_file="preprocessing_log_arg_chinese.txt"
    )

    arg_network_translator = PreprocessingTranslator(
       src_lang="en",
       tokenizer_name="DAMO-NLP-MT/polylm-13b",
       log_file="preprocessing_log_arg.txt"
    )

    fang_covid_translator = PreprocessingTranslator(
       credentials_path=CREDENTIALS_PATH,
       tokenizer_name="DAMO-NLP-MT/polylm-13b",
       src_lang="de",
       tgt_lang="en",
       log_file="preprocessing_log_fang_covid.txt"
    )

    ban_fake_news_translator = PreprocessingTranslator(
       credentials_path=CREDENTIALS_PATH,
       tokenizer_name="DAMO-NLP-MT/polylm-13b",
       src_lang="bn",
       tgt_lang="en",
    )
    
    target_count = 6000

    #Load the DataFrame from the specified file paths
    fang_covid_translator.load_json_files(FOLDER_PATH_FANG_COVID)
    ban_fake_news_translator.load_dataframe_from_csv(file_paths_ban_fake_news)
    arg_network_translator.load_dataframe_from_json(file_paths_arg_dataset)
    
    afnd_translator.load_dataframe_from_csv(AFND_PATH)
    
    arg_network_chinese_translator.load_dataframe_from_json(file_paths_arg_dataset_chinese)
    

    #Preprocess the DataFrame, translate the content of the specified column and save the translated DataFrame
    fang_covid_translator.preprocess_and_translate(column_name="article", target_count=target_count, 
                                                  source_column="article", target_column="translated_article", save_path=FANG_COVID_SAVE_PATH)
    ban_fake_news_translator.preprocess_and_translate(column_name="content", target_count=target_count,
                                                       source_column="content", target_column="translated_content", save_path=BAN_FAKE_NEWS_SAVE_PATH)
    afnd_translator.preprocess_and_translate(target_count=target_count,target_column="translated_content", source_column="content", save_path=AFND_SAVE_PATH, column_name="content")
    arg_network_chinese_translator.preprocess_and_translate(target_count=target_count,target_column="translated_content", source_column="content", save_path=ARG_SAVE_PATH_CHINESE, column_name="content")



    #Only Preprocess the DataFrame without translation
    arg_network_translator.preprocess_dataframe(column_name="content", target_count=target_count)
    arg_network_translator.write_dataframe_to_csv(ARG_SAVE_PATH)
    
    

        
    

