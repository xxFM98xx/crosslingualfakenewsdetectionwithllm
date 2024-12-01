import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from string import Template
import re
import torch
from torch.utils.data import DataLoader, Dataset
from .ModelInputOutputHandlers import get_task
from .ModelInputOutputHandlers import ModelHandler

class TranslationDataset(Dataset):
    """
    Custom Dataset class for translations.

    Attributes:
        data (pd.DataFrame): The dataset containing the text data.
        source_column (str): The column name of the text to be translated.
        template (Template): The template used for generating translation prompts.
        source_language (str): The source language of the input text.
        model_name (str): The name of the model used for translation.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
    """

    def __init__(self, data: pd.DataFrame, source_column: str, template: Template, source_language: str,model_name:str, tokenizer:AutoTokenizer):
        self.data = data
        self.source_column = source_column
        self.template = template
        self.language = source_language
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.task = get_task("translation")
        self.model_handler = ModelHandler._get_model_handler(model_name = self.model_name, task = self.task, tokenizer = self.tokenizer,language=self.language)
        
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text = self.data.iloc[idx][self.source_column]
        prompt = self.model_handler.format_prompt(template = self.template, input_text = input_text, source_language = self.language)
        return prompt

class PreTranslationsModule:
    """
    This class is responsible for translating input text to the target language using a large language model.

    Attributes:
        dataset (pd.DataFrame): The dataset containing the input text.
        model (AutoModelForCausalLM): The large language model used for translation.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        source_language (str): The source language of the input text.
        batch_size (int): The size of batches for processing translations.
        num_workers (int): The number of workers for data loading.
        max_new_tokens (int): The maximum number of tokens to generate for each translation.
    """

    def __init__(self, dataset: pd.DataFrame, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, source_language: str, model_name:str, batch_size: int = 8, num_workers: int = 0, max_new_tokens:int = 500):
        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer
        self.language = source_language
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = num_workers
        self.template = self._init_template()
        self.model_name = model_name
        self.task = get_task("translation")
        self.model_handler = ModelHandler._get_model_handler(model_name = self.model_name, task = self.task, tokenizer = self.tokenizer,language=self.language)
        self.max_new_tokens = max_new_tokens
        
        
    def translate_batch(self, inputs):
        """
        Translates a batch of inputs using the model.

        Args:
            inputs (list): List of text inputs to be translated.

        Returns:
            List[str]: The translated texts.
        """
        tokenized_inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to(self.device)
        print(f"tokenized_inputs: {tokenized_inputs}")
        generate_ids = self.model.generate(tokenized_inputs.input_ids, attention_mask=tokenized_inputs.attention_mask, do_sample=True, max_new_tokens= self.max_new_tokens, top_k=0,temperature=0.1, top_p=0.5)
        batch_decoded_output= self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(f"batch_decoded_output: {batch_decoded_output}", flush=True)
        generated_only = []
        for full_output in batch_decoded_output:
            generated_text = self.model_handler.extract_reply(full_output)
            generated_only.append(generated_text)
        print(f"Generated only in generate_rationale_batch: {generated_only}\n",flush=True)
        return generated_only
    
    def translate_all(self, source_column: str):
        """
        Translates all texts in the dataset.

        Args:
            source_column (str): The column containing the text to be translated.

        Returns:
            pd.DataFrame: The dataframe with the translated texts.
        """
        dataset = TranslationDataset(data=self.dataset, source_column=source_column, template= self.template, source_language = self.language, model_name = self.model_name, tokenizer = self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=lambda x: x)

        translated_texts = []
        for batch in dataloader:
            translated_texts.extend(self.translate_batch(batch))

        self.dataset[f"llm_translated_{source_column}"] = translated_texts
        return self.dataset

    def _init_template(self):
        """
        Initializes the translation template.

        Returns:
            Template: The template for translation prompts.
        """
        template = Template(
        """
        Translate the following text from $source_language into English. Return only the English translation without any additional text.
        
        Input:
        $input_text

        Translation:
        """
        )
        return template
        
        
        
