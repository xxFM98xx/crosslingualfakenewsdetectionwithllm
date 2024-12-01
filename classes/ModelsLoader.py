import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
import socket
import os
import numpy as np

class ModelsLoader:
    """
    A utility class for loading and setting up models with optional quantization.
    """

    def __init__(self, model_name: str, quantized: bool = False, distributed: bool = False):
        self.model_name = model_name
        self.quantized = quantized
        self.distributed = distributed

    def set_global_seed(self, seed: int):
        """
        Sets the seed for for reproducibility.

        Args:
            seed (int): The seed value to set.
        """
        np.random.seed(seed)
        set_seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)   
             
             
    def load_model(self):
        """
        Loads the specified model with optional quantization.

        Returns:
            tokenizer: The tokenizer associated with the model.
            model: The loaded model.
        """
        quantization_config = BitsAndBytesConfig(load_in_8bit=self.quantized)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left')
        

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.pad_token = tokenizer.eos_token
            print(f"The tokenizer.pad_token set as a {tokenizer.eos_token}")
        print("Tokenizer loaded.")
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            quantization_config=quantization_config if self.quantized else None,
                torch_dtype=torch.bfloat16
        )

        if self.distributed and torch.cuda.device_count() > 1:
            model = torch.nn.parallel.DistributedDataParallel(model)
            print(f"Model wrapped in DistributedDataParallel with {torch.cuda.device_count()} GPUs.")
        else:
            model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Model loaded on device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

        model.eval()
        return tokenizer, model
