import pandas as pd
from .PreTranslationsModule import PreTranslationsModule
from .RationaleModule import RationaleGenerator
from .InformationExtractor import InformationExtractor
from typing import List, Tuple
import os

def merge_rationale_dataframes(source_rationales_df: pd.DataFrame, 
                               translated_google_rationales_df: pd.DataFrame, 
                               translated_llm_rationales_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the three dataframes (source, Google-translated, and LLM-translated) while renaming the 
    cot_linguistic_style and cot_commonsense columns to keep track of their origin.
    
    Parameters:
    - source_rationales_df: The dataframe containing rationales in the source language.
    - translated_google_rationales_df: The dataframe containing rationales from Google-translated text.
    - translated_llm_rationales_df: The dataframe containing rationales from LLM-translated text.
    
    Returns:
    - A merged dataframe that combines all three with unique column names for cot_linguistic_style and cot_commonsense.
    """
    
    # Rename columns for source rationales
    source_rationales_df = source_rationales_df.rename(columns={
        "cot_linguistic_style": "cot_linguistic_style_source",
        "cot_commonsense": "cot_commonsense_source"
    })
    
    # Rename columns for Google-translated rationales
    translated_google_rationales_df = translated_google_rationales_df.rename(columns={
        "cot_linguistic_style": "cot_linguistic_style_google",
        "cot_commonsense": "cot_commonsense_google"
    })
    
    # Rename columns for LLM-translated rationales
    translated_llm_rationales_df = translated_llm_rationales_df.rename(columns={
        "cot_linguistic_style": "cot_linguistic_style_llm",
        "cot_commonsense": "cot_commonsense_llm"
    })
    
    # Merge all three dataframes on the common columns (assuming they share a common identifier)
    merged_df = source_rationales_df.merge(translated_google_rationales_df, on='id', suffixes=('', '_google')).merge(
        translated_llm_rationales_df, on='id', suffixes=('', '_llm'))

    return merged_df

class Pipeline:
    """
    A class to create a pipeline for generating rationales based on the dataset in the original language and the translated languages.
    
    Parameters:
    - dataset: The dataset containing the text data.
    - source_column: The name of the column containing the source text.
    - model: The model used for generating rationales and translation.
    - tokenizer: The tokenizer used for tokenizing in the creation of rationales and translation process.
    
    - source_language: The language of the source text.
    - perspectives: The perspectives used for generating rationales.
    - class_batch_size: The batch size used for generating rationales and translation in the individual classes.
    - save_path_dir: The directory where the processed datasets will be saved.
    - job_name: The name of the job used for saving the processed datasets.
    - num_workers: The number of workers used for parallel processing.
    - model_name: The name of the model used for generating rationales.
    - template_names: The names of the templates used for generating rationales.
    - max_new_tokens_translate: The maximum number of tokens to generate for each translation.
    - max_new_tokens_rationale: The maximum number of tokens to generate for each rationale.
    """
    def __init__(self, dataset: pd.DataFrame, source_column: str, model, tokenizer, source_language: str, perspectives: List[str], class_batch_size: int, num_workers: int=0,model_name:str=None, template_names: list[str] = None, save_path_dir:str=None, job_name:str=None, max_new_token_translate=500 ,max_new_tokens_rationale=500) -> None:
        """
        Initialize the Pipeline with the given parameters.
        """
        self.dataset = dataset
        allowed_languages = ['german', 'bengali', 'english','arabic','chinese']
        if source_language not in allowed_languages:
            raise ValueError(f"source_language must be one of {allowed_languages}, got '{source_language}' instead.")
        self.source_column = source_column
        self.model = model
        self.tokenizer = tokenizer
        self.source_language = source_language
        self.perspectives = perspectives
        self.class_batch_size = class_batch_size
        self.num_workers = num_workers
        if save_path_dir is not None:
            self.save_path_dir = save_path_dir
        else:
            raise ValueError("save_path_dir must be provided.")
        if job_name is not None:
            self.job_name = job_name
        else:
            raise ValueError("job_name must be provided.")
        
        if model_name is not None:
            self.model_name = model_name
        else:
            raise ValueError("model_name must be provided.")
        
        available_templates = ["cot", "step_back", "analogical_reasoning", "thot"]
        if template_names is None:
            self.template_names = available_templates

        elif isinstance(template_names, str):
            template_names = [template_names]
        if set(template_names).issubset(available_templates):
            self.template_names = template_names
        else:
            raise ValueError(f"Invalid template names. Available templates: {available_templates}")
        
        self.max_new_tokens_rationale = max_new_tokens_rationale
        self.max_new_tokens_translate = max_new_token_translate
        


    def pretranslate_generate_rationale(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        The function translates the dataset into English and generates rationales based on the dataset in the original language as well as in English.

        :return:
        - translated_llm_rationales_df: The dataframe containing the translated dataset (PretranslationsModule) with rationales generated from the translated dataset using the large language model.
        - translated_google_rationales_df: The dataframe containing the translated dataset (Google Translator) with rationales generated from the translated dataset using the Google Translator.
        - source_rationales_df: The dataframe containing the original dataset with rationales generated from the original dataset.
        """
        translation_completed = False
        translated_llm_rationales_df = pd.DataFrame()
        translated_google_rationales_df = pd.DataFrame()
        source_rationales_df = pd.DataFrame()

        if not translation_completed:
            pretranslation_module = PreTranslationsModule(dataset=self.dataset.copy(), model=self.model, tokenizer= self.tokenizer, source_language= self.source_language, batch_size= self.class_batch_size, num_workers=self.num_workers, model_name=self.model_name, max_new_tokens=self.max_new_tokens_translate)
            translated_llm_df = pretranslation_module.translate_all(self.source_column)

            information_extractor = InformationExtractor(df=translated_llm_df, translation_column_names=[f"llm_translated_{self.source_column}"],perspectives=self.perspectives)
            extracted_llm_translated_df = information_extractor.extract_all_translations()

            rationale_generator_en_llm = RationaleGenerator(extracted_llm_translated_df, self.model, self.tokenizer, self.perspectives, source_language="english", batch_size=self.class_batch_size, num_workers=self.num_workers,model_name=self.model_name, template_names=self.template_names, max_new_tokens=self.max_new_tokens_rationale)
            translated_llm_rationales_df = rationale_generator_en_llm.generate_all_rationales(f'llm_translated_{self.source_column}_extracted_translation')


        rationale_generator_en_google = RationaleGenerator(self.dataset.copy(), self.model, self.tokenizer, self.perspectives, source_language="english", batch_size=self.class_batch_size, num_workers=self.num_workers,model_name=self.model_name, template_names=self.template_names)

        if self.source_language.lower() == "german":
            rationale_generator_ger = RationaleGenerator(self.dataset.copy(), self.model, self.tokenizer, self.perspectives, source_language="german", batch_size=self.class_batch_size, num_workers=self.num_workers,model_name=self.model_name, template_names=self.template_names, max_new_tokens=self.max_new_tokens_rationale)
            source_rationales_df = rationale_generator_ger.generate_all_rationales(self.source_column)
        elif self.source_language.lower() == "bengali":
            rationale_generator_ben = RationaleGenerator(self.dataset.copy(), self.model, self.tokenizer, self.perspectives, source_language="bengali", batch_size=self.class_batch_size, num_workers=self.num_workers,model_name=self.model_name, template_names=self.template_names, max_new_tokens=self.max_new_tokens_rationale)
            source_rationales_df = rationale_generator_ben.generate_all_rationales(self.source_column)
        elif self.source_language.lower() == "arabic":
            rationale_generator_ar = RationaleGenerator(self.dataset.copy(), self.model, self.tokenizer, self.perspectives, source_language="arabic", batch_size=self.class_batch_size, num_workers=self.num_workers,model_name=self.model_name, template_names=self.template_names, max_new_tokens=self.max_new_tokens_rationale)
            source_rationales_df = rationale_generator_ar.generate_all_rationales(self.source_column)
        elif self.source_language.lower() == "chinese":
            rationale_generator_chi = RationaleGenerator(self.dataset.copy(), self.model, self.tokenizer, self.perspectives, source_language="chinese", batch_size=self.class_batch_size, num_workers=self.num_workers,model_name=self.model_name, template_names=self.template_names, max_new_tokens=self.max_new_tokens_rationale)
            source_rationales_df = rationale_generator_chi.generate_all_rationales(self.source_column)
        
        translated_google_rationales_df = rationale_generator_en_google.generate_all_rationales("translated_" + self.source_column)

   
        translated_llm_rationales_df.to_csv(f"{self.save_path_dir}{self.job_name}_llm_processed_rationale.csv", index=False)
        translated_google_rationales_df.to_csv(f"{self.save_path_dir}{self.job_name}_google_processed_rationale.csv", index=False)
        source_rationales_df.to_csv(f"{self.save_path_dir}{self.job_name}_processed_rationale.csv", index=False)
        print("Generate rationales with pre-translation completed.")
        rationales_df = merge_rationale_dataframes(source_rationales_df, translated_google_rationales_df, translated_llm_rationales_df)
        rationales_df.to_csv(f"{self.save_path_dir}{self.job_name}.csv", index=False)
        return rationales_df
    
    def generate_rationales_without_translation(self) -> pd.DataFrame:
        """
        Generates rationales for the dataset directly in its original language without any pre-translation.

        :return:
        - source_rationales_df: The dataframe containing the original dataset with generated rationales.
        """

        source_rationales_df = pd.DataFrame()
        completed = False

        if not completed:
            rationale_generator = RationaleGenerator(self.dataset.copy(), self.model, self.tokenizer, self.perspectives, source_language=self.source_language, batch_size=self.class_batch_size, num_workers=self.num_workers,model_name=self.model_name, template_names=self.template_names, max_new_tokens=self.max_new_tokens_rationale)
            print(f"Starting rationale generation for source language: {self.source_language} without translation...")
            source_rationales_df = rationale_generator.generate_all_rationales(self.source_column)


        source_rationales_df.to_csv(f"{self.save_path_dir}{self.job_name}_processed_rationale.csv", index=False)
        return source_rationales_df
      
        
        
        
        
        
        
        
        
        
        
        
        

        
        
