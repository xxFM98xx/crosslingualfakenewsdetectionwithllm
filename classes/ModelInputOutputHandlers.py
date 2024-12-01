from abc import ABC, abstractmethod
from string import Template

def remove_special_tokens(text, tokenizer):
    special_tokens = tokenizer.all_special_tokens + [
        '<', '</s>', 'system', 'user', 'assistant',
        '[INST]', '[/INST]', '<<SYS>>', '<</SYS>>'
        ]
    for token in special_tokens:
        text = text.replace(token, '')
    return text.strip()

def remove_prefix(text:str):
    if text.startswith("model"):
            return text[len("model"):].strip()
    elif text.startswith("assistant"):
        return text[len("assistant"):].strip()
    return text

class Task(ABC):
    """
    Abstract base class for defining a task.
    """
    @abstractmethod
    def format_prompt(self, template: Template, **kwargs):
        """
        Formats the prompt for the task.
        """
        pass

    @abstractmethod
    def extract_reply(self, full_output: str, tokenizer):
        """
        Extracts the reply from the full output.
        """
        pass

class FakeNewsDetectionTask(Task):
    """
    Task class for fake news detection
    
    Attributes:
        SYSTEM_MESSAGES (dict): A dictionary containing system messages in different languages.
    """
    SYSTEM_MESSAGES = {
            "english": "You are an expert in fake news detection.",
            "german": "Sie sind ein Experte für die Erkennung von Falschnachrichten.",
            "bengali": "আপনি একজন ভুয়া খবর সনাক্তকরণ বিশেষজ্ঞ।", 
            "arabic": "أنت خبير في اكتشاف الأخبار الكاذبة.", 
            "chinese": "您是假新闻检测方面的专家",
            }
    def format_prompt(self, template: Template, item: str, perspective: str):
        """
        Formats the prompt for the fake news detection task.
        
        Args:
            template (Template): The template for the prompt.
            item (str): The news item to be evaluated.
            perspective (str): The perspective from which the news item is to be evaluated.
        
        Returns:
            str: The formatted prompt.
        """
        return template.substitute(news_item=item, perspective=perspective).strip()

    def extract_reply(self, full_output: str, tokenizer):
        """
        Extracts the reply from the full output.
        
        Args:
            full_output (str): The full output text.
            tokenizer: The tokenizer associated with the model.
        
        Returns:
            str: The extracted reply.
        """
        
        return remove_special_tokens(full_output.strip(), tokenizer)
    def get_system_message(self, language: str) -> str:
        return self.SYSTEM_MESSAGES.get(language, "")

class TranslationTask(Task):
    """
    Task class for translation
    
    Attributes:
        SYSTEM_MESSAGES (dict): A dictionary containing system messages in different languages.
    """
    SYSTEM_MESSAGES = {
    "english": "You are an expert in translating texts.",
    "german":  "Du bist ein Experte beim Übersetzen von Texten.",
    "bengali":  "আপনি পাঠ্য অনুবাদে বিশেষজ্ঞ।",
    "arabic":  "أنت خبير في ترجمة النصوص.",
    "chinesisch": "您是翻译文本的专家。"
    }
    def format_prompt(self, template: Template, input_text: str, source_language: str):
        """
        Formats the prompt for the translation task.
        
        Args:
            template (Template): The template for the prompt.
            input_text (str): The text to be translated.
            source_language (str): The source language of the text.
        
        Returns:
            str: The formatted prompt.
        """
        return template.substitute(input_text=input_text, source_language=source_language).strip()

    def extract_reply(self, full_output: str, tokenizer):
        """
        Extracts the reply from the full output.
        
        Args:
            full_output (str): The full output text.
            tokenizer: The tokenizer associated with the model.
        
        Returns:
            str: The extracted reply.
        """
        without_special_tokens = remove_special_tokens(full_output.strip(), tokenizer)
        without_prefix = remove_prefix(without_special_tokens)
        return without_prefix
    
    def get_system_message(self, language: str) -> str:
        """
        Returns the system message for the specified language.
        
        Args:
            language (str): The language for which the system message is needed.
        
        Returns:
            str: The system message for the specified language.
        """
        
        return self.SYSTEM_MESSAGES.get(language, "")
    
def get_task(task_name: str) -> Task:
    """
    Returns the task object based on the provided task name.
    
    Args:
        task_name (str): The name of the task.
        
    Returns:
        Task: The task object corresponding to the task name.

    Raises:
        ValueError: If the task name is unknown.
    """
    if task_name == "fake_news_detection":
        return FakeNewsDetectionTask()
    elif task_name == "translation":
        return TranslationTask()
    else:
        raise ValueError(f"Unknown task: {task_name}")
    
class ModelHandler(ABC):
    """
    Abstract base class for defining a model handler.
    
    Attributes:
        tokenizer: The tokenizer associated with the model.
        language (str): The language for the model handler.
        task (Task): The specific task for which the model handler is needed.
    """
    def __init__(self, tokenizer, language, task:Task):
        self.tokenizer = tokenizer
        self.language = language
        self.task = task

    @abstractmethod
    def format_prompt(self, template: Template, **kwargs):
        return self.task.format_prompt(template,**kwargs)
    
    
    @abstractmethod
    def extract_reply(self, full_output: str):
        return self.task.extract_reply(full_output, self.tokenizer)

    @staticmethod
    def _get_model_handler(model_name, tokenizer, language, task):
        """
        Returns the appropriate model handler based on the provided model name.
        Args:
            model_name (str): The name of the model to be used.
            tokenizer: The tokenizer associated with the model.
            language (str): The language for the model handler.
            task (str): The specific task for which the model handler is needed.
        Returns:
            An instance of the appropriate model handler class based on the model name.
            If the model name does not match any specific handler, a DefaultModelHandler is returned.
        """

        if "SeaLLMs/SeaLLMs-v3-7B-Chat" in model_name:
            return SeaLLMsHandler(tokenizer, language, task)
        elif "meta-llama/Llama-2-7b-chat-hf" in model_name:
            return Llama2ChatHandler(tokenizer, language, task)
        elif "meta-llama/Meta-Llama-3.1-8B-Instruct" in model_name:
            return Llama31InstructHandler(tokenizer, language, task)
        elif "google/gemma-1.1-7b-it" in model_name:
            return GemmaHandler(tokenizer, language, task)
        elif "Qwen/Qwen2-7B-Instruct" in model_name:
            return Qwen2Handler(tokenizer, language, task)
        elif "DAMO-NLP-MT/polylm-chat-13b" in model_name:
            return PolyLMHandler(tokenizer, language, task)
        elif "FreedomIntelligence/phoenix-inst-chat-7b" in model_name:
            return PhoenixHandler(tokenizer, language, task)
        elif "tiiuae/falcon-7b-instruct" in model_name:
            return Falcon1Handler(tokenizer, language, task)
        else:
            return DefaultModelHandler(tokenizer, language, task)
    

class SeaLLMsHandler(ModelHandler):
    """
    Model handler for SeaLLMs
    
    Attributes:
        tokenizer: The tokenizer associated with the model.
        language (str): The language for the model handler.
        task (Task): The specific task for which the model handler is needed.
    
    SeaLLMs requires a specific format for the prompt, which includes system and user messages.
    """
    def format_prompt(self, template: Template, **kwargs):

        system_message = self.task.get_system_message(self.language)


        # Construct the messages structure expected by SeaLLMs
        user_message = self.task.format_prompt(template, **kwargs).strip()
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        # Apply chat template using SeaLLMs tokenizer
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    def extract_reply(self, full_output: str):
        # Special tokens
        assistant_token = "assistant"
        assistant_end_token = "<|im_end|>"

        assistant_start = full_output.find(assistant_token)
        if assistant_start != -1:
            assistant_start += len(assistant_token)
            assistant_end = full_output.find(assistant_end_token, assistant_start)
            if assistant_end != -1:
                assistant_reply = full_output[assistant_start:assistant_end].strip()
            else:
                assistant_reply = full_output[assistant_start:].strip()
            return self.task.extract_reply(assistant_reply, self.tokenizer)
        else:
            return self.task.extract_reply(full_output.strip(), self.tokenizer)
        
class Llama2ChatHandler(ModelHandler):
    """
    Model handler for Llama 2
    
    Attributes:
        tokenizer: The tokenizer associated with the model.
        language (str): The language for the model handler.
        task (Task): The specific task for which the model handler is needed.
    
    Llama 2 requires a specific format for the prompt, which includes system and user messages.
    """
    def format_prompt(self, template: Template, **kwargs):
        B_INST = "[INST]"
        E_INST = "[/INST]"
        B_SYS =  "<<SYS>>\n"
        E_SYS =  "\n<</SYS>>\n\n"
        
        system_message = self.task.get_system_message(self.language)

        user_message = self.task.format_prompt(template, **kwargs)

        if system_message:
            combined_message = f"{B_SYS}{system_message}{E_SYS}{user_message}"
        else:
            combined_message = user_message
        
        prompt = f"{B_INST} {combined_message} {E_INST}"
        
        return prompt

    def extract_reply(self, full_output: str):
        E_INST = "[/INST]"
        reply_start = full_output.rfind(E_INST)
        if reply_start != -1:
            reply_start += len(E_INST)
            assistant_reply = full_output[reply_start:].strip()
            assistant_reply = self.task.extract_reply(assistant_reply, self.tokenizer)
            return assistant_reply
        else:
            assistant_reply = self.task.extract_reply(full_output, self.tokenizer)
            return assistant_reply


class GemmaHandler(ModelHandler):
    """
    Model handler for Gemma
    
    Attributes:
        tokenizer: The tokenizer associated with the model.
        language (str): The language for the model handler.
        task (Task): The specific task for which the model handler is needed.
    
    Gemma requires a specific format for the prompt, which includes system and user messages.
    """
    def format_prompt(self, template: Template, **kwargs):
        user_message = self.task.format_prompt(template, **kwargs).strip()
        chat = [
            {"role": "user", "content": user_message},
        ]
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        return prompt

    def extract_reply(self, full_output: str):
        model_token = "<start_of_turn>model\n"
        end_token = "<end_of_turn>"

        reply_start = full_output.find(model_token)
        if reply_start != -1:
            reply_start += len(model_token)
            reply_end = full_output.find(end_token, reply_start)
            if reply_end != -1:
                assistant_reply = full_output[reply_start:reply_end].strip()
            else:
                assistant_reply = full_output[reply_start:].strip()
            return self.task.extract_reply(assistant_reply, self.tokenizer)
        else:
            return self.task.extract_reply(full_output.strip(), self.tokenizer)
        
class Llama31InstructHandler(ModelHandler):
    """
    Model handler for Llama 3.1
    
    Attributes:
        tokenizer: The tokenizer associated with the model.
        language (str): The language for the model handler.
        task (Task): The specific task for which the model handler is needed.
    
    Llama 3.1 requires a specific format for the prompt, which includes system and user messages.
    """
    
    def format_prompt(self, template: Template, **kwargs):
        system_message = self.task.get_system_message(self.language)

        user_message = self.task.format_prompt(template, **kwargs)
        prompt = "<|begin_of_text|>\n"
        
        if system_message:
            prompt += f"<|start_header_id|>system<|end_header_id|>\n{system_message}\n<|eot_id|>\n"
        
        prompt += f"<|start_header_id|>user<|end_header_id|>\n{user_message}\n<|eot_id|>\n"
        
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
        
        return prompt

    def extract_reply(self, full_output: str):    
        assistant_header = "<|start_header_id|>assistant<|end_header_id|>\n"
        end_header = "<|eot_id|>"
        reply_start = full_output.find(assistant_header)
        if reply_start != -1:
            reply_start += len(assistant_header)
            reply_end = full_output.find(end_header, reply_start)
            
            if reply_end != -1:
                assistant_reply = full_output[reply_start:reply_end].strip()
                return self.task.extract_reply(assistant_reply, self.tokenizer)
                
            else:
                assistant_reply = full_output[reply_start:].strip()
                return self.task.extract_reply(assistant_reply, self.tokenizer)
        else:
            return self.task.extract_reply(full_output.strip(), self.tokenizer)
        
class Qwen2Handler(ModelHandler):
    """
    Model handler for Qwen2
    
    Attributes:
        tokenizer: The tokenizer associated with the model.
        language (str): The language for the model handler.
        task (Task): The specific task for which the model handler is needed.
    
    Qwen2 requires a specific format for the prompt, which includes system and user messages.
    """
    
    def format_prompt(self, template: Template, **kwargs):
        system_message = self.task.get_system_message(self.language)


        # Construct the messages structure expected by Qwen2
        user_message = self.task.format_prompt(template, **kwargs).strip()
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        # Apply chat template using Qwen2 tokenizer
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    def extract_reply(self, full_output: str):
        assistant_header = "assistant"
        end = "<|im_end|>"
        reply_start = full_output.find(assistant_header)
        if reply_start != -1:
            print(f"reply_start found at {reply_start}",flush=True)
            reply_start += len(assistant_header)
            reply_end = full_output.find(end, reply_start)
            if reply_end != -1:
                    print(f"reply_end found at {reply_end}",flush=True)
                    assistant_reply = full_output[reply_start:reply_end].strip()
            else:
                print(f"reply_end not found",flush=True)
                assistant_reply = full_output[reply_start:].strip()
            return self.task.extract_reply(assistant_reply, self.tokenizer)
        else:
            print(f"reply_start not found",flush=True)
            return self.task.extract_reply(full_output.strip(), self.tokenizer)
        
class PolyLMHandler(ModelHandler):
    """
    Model handler for PolyLM
    
    Attributes:
        tokenizer: The tokenizer associated with the model.
        language (str): The language for the model handler.
        task (Task): The specific task for which the model handler is needed.
    
    PolyLM requires a specific format for the prompt, which includes system and user messages.
    """
    def format_prompt(self, template: Template, **kwargs):
        system_message = self.task.get_system_message(self.language)


        user_message = self.task.format_prompt(template, **kwargs).strip()

        prompt = f"<|user|>\n{user_message}\n<|assistant|>\n"
        return prompt
  
    def extract_reply(self, full_output: str): 
        start_token = "\n<|assistant|>\n"
        end_token = "<p>"

        assistant_start = full_output.find(start_token)
        if assistant_start != -1:
            assistant_start += len(start_token)
            assistant_end = full_output.find(end_token, assistant_start)
            if assistant_end != -1:
                assistant_reply = full_output[assistant_start:assistant_end].strip()
            else:
                assistant_reply = full_output[assistant_start:].strip()
            return self.task.extract_reply(assistant_reply, self.tokenizer)
        else:
            return self.task.extract_reply(full_output.strip(), self.tokenizer)
class PhoenixHandler(ModelHandler):
    """
    Model handler for Phoenix
    
    Attributes:
        tokenizer: The tokenizer associated with the model.
        language (str): The language for the model handler.
        task (Task): The specific task for which the model handler is needed.
    
    Phoenix requires a specific format for the prompt, which includes system and user messages.
    """
    
    def format_prompt(self, template: Template, **kwargs):
        system_message = self.task.get_system_message(self.language)


        user_message = self.task.format_prompt(template, **kwargs).strip()

        prompt = (
            f"{system_message}\n\n"
            f"Human: <s>{user_message}</s>\n"
            f"Assistant: <s>"
        )

        return prompt

    def extract_reply(self, full_output: str):
        assistant_prefix = "Assistant:"
        assistant_suffix = "</s>"

        assistant_start = full_output.find(assistant_prefix)
        if assistant_start != -1:
            assistant_start += len(assistant_prefix)
            assistant_end = full_output.find(assistant_suffix, assistant_start)
            if assistant_end != -1:
                assistant_reply = full_output[assistant_start:assistant_end].strip()
            else:
                assistant_reply = full_output[assistant_start:].strip()
            return self.task.extract_reply(assistant_reply, self.tokenizer)
        else:
            return self.task.extract_reply(full_output.strip(), self.tokenizer)

class Falcon1Handler(ModelHandler):
    """
    Model handler for Falcon 1
    
    Attributes:
        tokenizer: The tokenizer associated with the model.
        language (str): The language for the model handler.
        task (Task): The specific task for which the model handler is needed.
        
    Falcon 1 requires a specific format for the prompt, which includes system and user messages.
    """
    def format_prompt(self, template: Template, **kwargs):
        system_message = self.task.get_system_message(self.language)


        user_message = self.task.format_prompt(template, **kwargs).strip()

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    def extract_reply(self, full_output: str):
        assistant_prefix = "Assistant:"
        eos_token = "<|endoftext|>"

        assistant_start = full_output.find(assistant_prefix)
        if assistant_start != -1:
            assistant_start += len(assistant_prefix)
            assistant_end = full_output.find(eos_token, assistant_start)
            if assistant_end != -1:
                assistant_reply = full_output[assistant_start:assistant_end].strip()
            else:
                assistant_reply = full_output[assistant_start:].strip()

            return self.task.extract_reply(assistant_reply, self.tokenizer)
        else:

            return self.task.extract_reply(full_output.strip(), self.tokenizer)



        
class DefaultModelHandler(ModelHandler):
    def format_prompt(self, template: Template, **kwargs):
        return self.task.format_prompt(template, **kwargs)

    def extract_reply(self, full_output: str):
        return self.task.extract_reply(full_output.strip(), self.tokenizer)
