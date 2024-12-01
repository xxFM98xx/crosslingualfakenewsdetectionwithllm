import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from string import Template
from .ModelInputOutputHandlers import  get_task
from .ModelInputOutputHandlers import ModelHandler


class RationaleDataset(Dataset):
    """
    Custom Dataset class for generating rationales.

    Attributes:
        data (pd.DataFrame): The dataset containing the text data.
        source_column (str): The column name of the text for which rationales are to be generated.
        templates (dict): Dictionary of templates for different rationale types.
        perspectives (list): List of perspectives for rationale generation.
        model_name (str): The name of the model used for generating rationales.
        language (str): The language of the text data.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
    """

    def __init__(self, data: pd.DataFrame, source_column: str, templates: dict, perspectives: list,model_name:str,language:str,tokenizer:AutoTokenizer):
        self.data = data
        self.source_column = source_column
        self.templates = templates
        self.perspectives = perspectives
        self.model_name = model_name
        self.language = language
        self.tokenizer = tokenizer
        self.task = get_task("fake_news_detection")
        self.model_handler = ModelHandler._get_model_handler(model_name = self.model_name, task = self.task, tokenizer = self.tokenizer,language=self.language)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        item = row[self.source_column]
        prompts = []
        for template_name, template in self.templates.items():
            for perspective in self.perspectives:
                if "translated_" in self.source_column:
                    prompt = self.model_handler.format_prompt(template = template, item = item, perspective = perspective["english"])
                    prompts.append((idx, template_name, perspective["english"], prompt))
                else:
                    prompt = self.model_handler.format_prompt(template = template, item = item, perspective = perspective["english"])
                    prompts.append((idx, template_name, perspective[self.language], prompt))
                
        print(f"Prompts in getitem: type: {type}\n",flush=True)
        return prompts

class RationaleGenerator:
    """
    Generates rationales from specific perspectives for input text in the dataset using a large language model.

    Attributes:
        dataset (pd.DataFrame): The dataset containing the input text.
        model (AutoModelForCausalLM): The large language model used for generating rationales.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        perspectives (list[str]): The perspectives for generating rationales.
        source_language (str): The source language of the input text.
        batch_size (int): The size of batches for processing rationales.
        num_workers (int): The number of workers for data loading.
        
        template_names (list[str]): The templates for generating rationales. Default is None and all templates available(cot, step_back, analogical_reasoning, thot) are used.
    """

    def __init__(self, dataset: pd.DataFrame, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, perspectives: list[str], source_language: str, batch_size: int = 8, num_workers: int = 0, template_names: list[str] = None,model_name:str=None,max_new_tokens=500) -> None:
        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer
        self.perspectives = perspectives
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = num_workers
        self.source_language = source_language
        if model_name is not None:
            self.model_name = model_name
        else:
            self.model_name = "unknown"
        self.task = get_task("fake_news_detection")
        
        # Initialize templates
        self.available_templates = ["cot", "step_back", "analogical_reasoning", "thot"]
        
        if template_names is None:
            self.templates = {
                "cot": self._init_template_cot(),
                "step_back": self._init_template_step_back_with_perspective(),
                "analogical_reasoning": self._init_template_analogical_reasoning_with_perspective(),
                "thot": self._init_template_thot_with_perspective()
            }
        # Check if the template names are valid and initialize the templates
        elif set(template_names).issubset(self.available_templates):
                self.templates = {name: self._init_template(name) for name in template_names}
        else:
                raise ValueError(f"Invalid template names. Available templates: {self.available_templates}")
        self.model_handler = ModelHandler._get_model_handler(model_name = self.model_name, task = self.task, tokenizer = self.tokenizer,language=self.source_language)
        self.max_new_tokens = max_new_tokens
        

    def generate_rationale_batch(self, prompts):
        """
        Generates rationales for a batch of prompts.

        Args:
            prompts (list): List of prompts to generate rationales for.

        Returns:
            List[str]: The generated rationales.
        """
        print("Entering generate_rationale_batch\n",flush=True)
        input_texts = [prompt for _, _, _, prompt in prompts]
        print(f"Input texts in generate_rationale_batch: {input_texts}\n",flush=True)
        tokenized_inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=1250).to(self.device)
        generate_ids = self.model.generate(tokenized_inputs.input_ids, attention_mask=tokenized_inputs.attention_mask, do_sample=True, max_new_tokens=self.max_new_tokens, top_k=0,temperature=0.1, top_p=0.5)
        print(f"Generate ids in generate_rationale_batch: {generate_ids}\n",flush=True)
        batch_decoded_output = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        print("full batch_decoded_output in generate_rationale_batch: ",batch_decoded_output,flush=True)
        print(f"Length of batch_decoded_output in generate_rationale_batch: {len(batch_decoded_output)}\n",flush=True)

        generated_only = []
        for full_output in batch_decoded_output:
            generated_text = self.model_handler.extract_reply(full_output)
            generated_only.append(generated_text)
        print(f"Generated only in generate_rationale_batch: {generated_only}\n",flush=True)
        return generated_only


    def generate_all_rationales(self, source_column: str):
        """
        Generates rationales for all texts in the dataset.

        Args:
            source_column (str): The column containing the text to generate rationales for.

        Returns:
            pd.DataFrame: The dataframe with the generated rationales.
        """
        print("Entering generate_all_rationales\n",flush=True)
        dataset = RationaleDataset(data = self.dataset, source_column=source_column, templates = self.templates, perspectives=self.perspectives,model_name=self.model_name,language=self.source_language,tokenizer=self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=lambda x: x)

        for batch in dataloader:
            if isinstance(batch[0], list):
                prompts = [item for sublist in batch for item in sublist]
            else:
                prompts = batch
            
            decoded_outputs = self.generate_rationale_batch(prompts)
            
            if len(prompts) != len(decoded_outputs):
                raise ValueError("Mismatch between number of prompts and decoded outputs.")

            for (index, template_name, perspective, _), decoded in zip(prompts, decoded_outputs):
                self.dataset.at[index, f"{template_name}_{perspective}"] = decoded

        return self.dataset
    
    def _init_template(self, template_name: str):
        template_methods = {
            "cot": self._init_template_cot,
            "step_back": self._init_template_step_back_with_perspective,
            "analogical_reasoning": self._init_template_analogical_reasoning_with_perspective,
            "thot": self._init_template_thot_with_perspective
        }
        if template_name in template_methods:
            return template_methods[template_name]()
        else:
            raise ValueError(f"Invalid template name: {template_name}")
        
    def _init_template_cot(self):
        cot_dict = {
        "english": Template(
        "You are an expert in fake news detection analyzing news from the perspective of $perspective. Here is a news item: '$news_item'\n"
        "Let's think step by step to determine whether this news is fake or not.\n\n"
        "## REASONING (from the perspective of $perspective): Please provide a detailed analysis in english language\n"
        "Finally, based on your analysis, determine whether the news item is fake or true.\n"
        "## IS THIS NEWS FAKE OR TRUE? Please state 'Fake' or 'True' \n"
    ),

        "german":Template(
            "Sie sind ein Experte für die Erkennung von Fake News und analysieren Nachrichten aus der Perspektive von $perspective. Hier ist ein Nachrichtenartikel: '$news_item'\n"
            "Lassen Sie uns Schritt für Schritt überlegen, ob diese Nachricht gefälscht ist oder nicht.\n\n"
            "## BEGRÜNDUNG (aus der Perspektive von $perspective): Bitte geben Sie eine ausführliche Analyse in deutscher Sprache \n"
            "Schließlich, basierend auf Ihrer Analyse, bestimmen Sie, ob der Nachrichtenartikel gefälscht oder wahr ist.\n"
            "## IST DIESE NACHRICHT FALSCH ODER WAHR? Bitte geben Sie 'Falsch' oder 'Wahr' an \n"
        ),

        "bengali":Template(
            "আপনি একজন ভুয়া খবর সনাক্তকরণ বিশেষজ্ঞ এবং $perspective এর দৃষ্টিকোণ থেকে খবর বিশ্লেষণ করছেন। এখানে একটি সংবাদ আইটেম রয়েছে: '$news_item'\n"
            "আসুন ধাপে ধাপে চিন্তা করি যে এই খবরটি ভুয়া কিনা তা নির্ধারণ করতে।\n\n"
            "## যুক্তি ( $perspective এর দৃষ্টিকোণ থেকে): দয়া করে বাংলা ভাষায় বিস্তারিত বিশ্লেষণ দিন \n"
            "শেষে, আপনার বিশ্লেষণের ভিত্তিতে নির্ধারণ করুন যে সংবাদ আইটেমটি ভুয়া না সত্য।\n"
            "## এই সংবাদটি ভুয়া না সত্য? অনুগ্রহ করে 'মিথ্যা' বা 'সত্য' উল্লেখ করুন \n"
        ),
        "arabic":Template(
            "أنت خبير في اكتشاف الأخبار المزيفة وتحليل الأخبار من منظور $perspective. إليك خبر: '$news_item'\n"
            "دعنا نفكر خطوة بخطوة لتحديد ما إذا كان هذا الخبر مزيفًا أم لا.\n\n"
            "## التعليل (من منظور $perspective): يرجى تقديم تحليل مفصل باللغة الإنجليزية\n"
            "أخيرًا، بناءً على تحليلك، حدد ما إذا كان الخبر مزيفًا أم حقيقيًا.\n"
            "## هل هذا الخبر مزيف أم حقيقي؟ يرجى تحديد \"مزيف\" أو \"حقيقي\" \n"
        ),
        "chinese": Template(
            "您是从$perspective的角度分析新闻的假新闻检测专家。这里有一条新闻：'$news_item'\n"
            "让我们一步步思考，判断这条新闻是否是假的。\n\n"
            "## 推理（从$perspective的角度）：请用英文提供详细分析\n"
            "最后，根据您的分析，判断这条新闻是假的还是真的。\n"
            "## 这则新闻是假的还是真的？请说出‘假’或‘真’\n"
        )
        } 
        return cot_dict[self.source_language]
    def _init_template_step_back_with_perspective(self):
        step_back_dict = {
            "english":Template(
                    "You are an expert in fake news detection analyzing news from the perspective of $perspective. Here is a news item: '$news_item'.\n"
                    "Step 1 - Abstraction: Instead of addressing whether this news is fake directly, let's first ask a step-back question to understand the high-level concept or principle from the perspective of $perspective.\n"
                    "## Step-Back Question: What are the main factors and context of this news item from the perspective of $perspective?\n"
                    "### Factors and Context: Factors and Context \n\n"
                    "Step 2 - Reasoning: Grounded on the facts regarding these main factors and context, let's think further from the perspective of $perspective. Is this news fake or not? Please provide your analysis with reasoning.\n"
                    "## REASONING (from the perspective of $perspective): Reasoning \n"
                    "## IS THIS NEWS FAKE OR TRUE?  True or Fake"
            ),

             "german":Template(
                    "Sie sind ein Experte für die Erkennung von Fake News und analysieren Nachrichten aus der Perspektive von $perspective. Hier ist ein Nachrichtenartikel: '$news_item'.\n"
                    "Schritt 1 - Abstraktion: Anstatt direkt zu beurteilen, ob diese Nachricht gefälscht ist, stellen wir zunächst eine Step-Back-Frage, um das übergeordnete Konzept oder Prinzip aus der Perspektive von $perspective zu verstehen.\n"
                    "## Step-Back-Frage: Was sind die Hauptfaktoren und der Kontext dieses Nachrichtenartikels aus der Perspektive von $perspective?\n"
                    "### Faktoren und Kontext: Faktoren und Kontext \n\n"
                    "Schritt 2 - Begründung: Basierend auf den Fakten zu diesen Hauptfaktoren und dem Kontext, denken wir weiter aus der Perspektive von $perspective. Ist diese Nachricht gefälscht oder nicht? Bitte geben Sie Ihre Analyse mit Begründung an.\n"
                    "## BEGRÜNDUNG (aus der Perspektive von $perspective): Begründung \n"
                    "## IST DIESE NACHRICHT FALSCH ODER WAHR? Falsch oder Wahr"
                ),

             "bengali":Template(
                    "আপনি একজন ভুয়া খবর সনাক্তকরণ বিশেষজ্ঞ এবং $perspective এর দৃষ্টিকোণ থেকে খবর বিশ্লেষণ করছেন। এখানে একটি সংবাদ আইটেম রয়েছে: '$news_item'.\n"
                    "পর্যায় ১ - বিমূর্তকরণ: সরাসরি এই খবরটি ভুয়া কিনা তা নির্ধারণ করার পরিবর্তে, আসুন প্রথমে একটি স্টেপ-ব্যাক প্রশ্ন করি যাতে $perspective এর দৃষ্টিকোণ থেকে উচ্চ-স্তরের ধারণা বা নীতিকে বোঝা যায়।\n"
                    "## স্টেপ-ব্যাক প্রশ্ন: $perspective এর দৃষ্টিকোণ থেকে এই সংবাদ আইটেমটির প্রধান কারণ এবং প্রেক্ষাপট কী?\n"
                    "### কারণ এবং প্রেক্ষাপট: কারণ এবং প্রেক্ষাপট \n\n"
                    "পর্যায় ২ - যুক্তি: এই প্রধান কারণ এবং প্রেক্ষাপট সম্পর্কে তথ্যের ভিত্তিতে, আসুন $perspective এর দৃষ্টিকোণ থেকে আরও চিন্তা করি। এই খবরটি ভুয়া নাকি সত্য? অনুগ্রহ করে আপনার বিশ্লেষণ যুক্তিসহ প্রদান করুন।\n"
                    "## যুক্তি ( $perspective এর দৃষ্টিকোণ থেকে): যুক্তি \n"
                    "## এই খবরটি কি ভুয়া না সত্য? ভুয়া বা সত্য"
                ),
             "arabic":Template(
                 "أنت خبير في اكتشاف الأخبار المزيفة وتحليل الأخبار من منظور المنظور. إليك خبر: '$news_item'.\n"
                "الخطوة 1 - التجريد: بدلاً من معالجة ما إذا كان هذا الخبر مزيفًا بشكل مباشر، دعنا أولاً نطرح سؤالاً للوراء لفهم المفهوم أو المبدأ رفيع المستوى من منظور المنظور.\n"
                "## سؤال للوراء: ما هي العوامل الرئيسية وسياق هذا الخبر من منظور المنظور؟\n"
                "### العوامل والسياق: العوامل والسياق \n\n"
                "الخطوة 2 - الاستدلال: بناءً على الحقائق المتعلقة بهذه العوامل الرئيسية والسياق، دعنا نفكر بشكل أعمق من منظور المنظور. هل هذا الخبر مزيف أم لا؟ يرجى تقديم تحليلك مع الاستدلال.\n"
                "## الاستدلال (من منظور المنظور): الاستدلال \n"
                "## هل هذا الخبر مزيف أم حقيقي؟ صحيح أم مزيف"
                 ),
             
             "chinese":Template(
                "您是从$perspective的角度分析新闻的假新闻检测专家。这里有一条新闻：'$news_item'。\n"
                "步骤1 - 抽象：我们先不直接讨论这条新闻是否是假的，而是先提出一个后退的问题，从$perspective的角度了解高层次的概念或原理。\n"
                "## 后退问题：从$perspective的角度来看，这条新闻的主要因素和背景是什么？\n"
                "### 因素和背景：因素和背景 \n\n"
                "步骤2 - 推理：基于这些主要因素和背景的事实，让我们从$perspective的角度进一步思考。这条新闻是假的吗？请提供您的分析和推理。\n"
                "## 推理（从$perspective的角度）：推理 \n"
                "## 这条新闻是假的还是真的？是真的还是假的"
             )
             
            }
        return step_back_dict[self.source_language]

    def _init_template_analogical_reasoning_with_perspective(self):
        analogical_reasoning_dict = {
        "english":Template(
                                                "You are an expert in fake news detection analyzing news from the perspective of $perspective. Here is a news item: '$news_item'.\n"
                                                "Step 1 - Self-Generated Exemplars: Instead of addressing whether this news is fake directly, let's first recall relevant news items and their analyses from the perspective of $perspective.\n"
                                                "## Recall relevant news items and solutions (from the perspective of $perspective):\n"
                                                "### Relevant News Item 1:\n"
                                                "#### Description: Describe News 1\n"
                                                "#### Analysis: Analyse News 1\n"
                                                "### Relevant News Item 2:\n"
                                                "#### Description: Describe News 2\n"
                                                "#### Analysis: Analyse News 2\n"
                                                "### Relevant News Item 3:\n"
                                                "#### Description: Describe News 3\n"
                                                "#### Analysis: Analyse News 3\n\n"
                                                "Step 2 - Analysis: Grounded on the facts and solutions of these relevant news items, identify common principles or patterns from the perspective of $perspective.\n"
                                                "## Common Principles or Patterns: Common Principles or Patterns\n\n"
                                                "Based on these principles or patterns, let's analyze the current news item.\n"
                                                "### REASONING (from the perspective of $perspective): Reasoning\n"
                                                "### IS THIS NEWS FAKE OR TRUE? True or Fake\n"
                                            ),

        "german":Template(
            "Sie sind ein Experte für die Erkennung von Fake News und analysieren Nachrichten aus der Perspektive von $perspective. Hier ist ein Nachrichtenartikel: '$news_item'.\n"
            "Schritt 1 - Selbstgenerierte Beispiele: Anstatt direkt zu beurteilen, ob diese Nachricht gefälscht ist, erinnern wir uns zunächst an relevante Nachrichtenartikel und deren Analysen aus der Perspektive von $perspective.\n"
            "## Erinnern Sie sich an relevante Nachrichtenartikel und Lösungen (aus der Perspektive von $perspective):\n"
            "### Relevanter Nachrichtenartikel 1:\n"
            "#### Beschreibung: Beschreiben Sie Nachrichtenartikel 1\n"
            "#### Analyse: Analysieren Sie Nachrichtenartikel 1\n"
            "### Relevanter Nachrichtenartikel 2:\n"
            "#### Beschreibung: Beschreiben Sie Nachrichtenartikel 2\n"
            "#### Analyse: Analysieren Sie Nachrichtenartikel 2\n"
            "### Relevanter Nachrichtenartikel 3:\n"
            "#### Beschreibung: Beschreiben Sie Nachrichtenartikel 3\n"
            "#### Analyse: Analysieren Sie Nachrichtenartikel 3\n\n"
            "Schritt 2 - Analyse: Basierend auf den Fakten und Lösungen dieser relevanten Nachrichtenartikel identifizieren wir gemeinsame Prinzipien oder Muster aus der Perspektive von $perspective.\n"
            "## Gemeinsame Prinzipien oder Muster: Gemeinsame Prinzipien oder Muster\n\n"
            "Basierend auf diesen Prinzipien oder Mustern analysieren wir den aktuellen Nachrichtenartikel.\n"
            "### BEGRÜNDUNG (aus der Perspektive von $perspective): Begründung\n"
            "### IST DIESE NACHRICHT FALSCH ODER WAHR? Falsch oder Wahr\n"
        ),
        "bengali":Template(
    "আপনি একজন ভুয়া খবর সনাক্তকরণ বিশেষজ্ঞ এবং $perspective এর দৃষ্টিকোণ থেকে খবর বিশ্লেষণ করছেন। এখানে একটি সংবাদ আইটেম রয়েছে: '$news_item'.\n"
    "পর্যায় ১ - স্বয়ং-উৎপন্ন উদাহরণ: সরাসরি এই খবরটি ভুয়া কিনা তা নির্ধারণ করার পরিবর্তে, আসুন প্রথমে $perspective এর দৃষ্টিকোণ থেকে প্রাসঙ্গিক খবর এবং তাদের বিশ্লেষণগুলি মনে করি।\n"
    "## প্রাসঙ্গিক খবর এবং সমাধানগুলি মনে করুন ( $perspective এর দৃষ্টিকোণ থেকে):\n"
    "### প্রাসঙ্গিক খবর ১:\n"
    "#### বর্ণনা: খবর ১ বর্ণনা করুন\n"
    "#### বিশ্লেষণ: খবর ১ বিশ্লেষণ করুন\n"
    "### প্রাসঙ্গিক খবর ২:\n"
    "#### বর্ণনা: খবর ২ বর্ণনা করুন\n"
    "#### বিশ্লেষণ: খবর ২ বিশ্লেষণ করুন\n"
    "### প্রাসঙ্গিক খবর ৩:\n"
    "#### বর্ণনা: খবর ৩ বর্ণনা করুন\n"
    "#### বিশ্লেষণ: খবর ৩ বিশ্লেষণ করুন\n\n"
    "পর্যায় ২ - বিশ্লেষণ: এই প্রাসঙ্গিক খবরগুলির তথ্য এবং সমাধানের ভিত্তিতে, $perspective এর দৃষ্টিকোণ থেকে সাধারণ নীতি বা প্যাটার্নগুলি সনাক্ত করুন।\n"
    "## সাধারণ নীতি বা প্যাটার্নগুলি: সাধারণ নীতি বা প্যাটার্নগুলি\n\n"
    "এই নীতি বা প্যাটার্নগুলির উপর ভিত্তি করে বর্তমান খবরটি বিশ্লেষণ করুন।\n"
    "### যুক্তি ( $perspective এর দৃষ্টিকোণ থেকে): যুক্তি\n"
    "### এই খবরটি কি ভুয়া না সত্য? ভুয়া বা সত্য\n"
    ),
    "arabic":Template(
        "أنت خبير في اكتشاف الأخبار المزيفة وتحليل الأخبار من منظور $perspective. إليك خبر: '$news_item'.\n"
        "الخطوة 1 - الأمثلة التي تم إنشاؤها ذاتيًا: بدلاً من معالجة ما إذا كان هذا الخبر مزيفًا بشكل مباشر، فلنتذكر أولاً الأخبار ذات الصلة وتحليلاتها من منظور $perspective.\n"
        "## تذكر الأخبار والحلول ذات الصلة (من منظور $perspective):\n"
        "### عنصر الأخبار ذات الصلة 1:\n"
        "#### الوصف: وصف الأخبار 1\n"
        "#### التحليل: تحليل الأخبار 1\n"
        "### عنصر الأخبار ذات الصلة 2:\n"
        "#### الوصف: وصف الأخبار 2\n"
        "#### التحليل: تحليل الأخبار 2\n"
        "### عنصر الأخبار ذات الصلة 3:\n"
        "#### الوصف: وصف الأخبار 3\n"
        "#### التحليل: تحليل الأخبار 3\n\n"
        "الخطوة 2 - التحليل: بناءً على الحقائق والحلول لهذه الأخبار ذات الصلة، حدد المبادئ أو الأنماط المشتركة من منظور $perspective.\n"
        "## المبادئ أو الأنماط المشتركة: المبادئ أو الأنماط المشتركة\n\n"
        "استنادًا إلى هذه المبادئ أو الأنماط، دعنا نحلل الخبر الحالي.\n"
        "### الاستدلال (من منظور $perspective): الاستدلال\n"
        "### هل هذا الخبر زائف أم حقيقي؟ صحيح أم مزيف\n"
        ),
    "chinese":Template(
        "您是从 $perspective 的角度分析新闻的虚假新闻检测专家。这里有一条新闻：“$news_item”。\n"
        "步骤 1 - 自我生成的样本：我们先不直接讨论这条新闻是否是假的，而是先从 $perspective 的角度回忆相关新闻及其分析。\n"
        "## 回忆相关新闻和解决方案（从 $perspective 的角度）：\n"
        "### 相关新闻 1：\n"
        "#### 描述：描述新闻 1\n"
        "#### 分析：分析新闻 1\n"
        "### 相关新闻 2：\n"
        "#### 描述：描述新闻 2\n"
        "#### 分析：分析新闻 2\n"
        "### 相关新闻 3：\n"
        "#### 描述：描述新闻 3\n"
        "#### 分析：分析新闻 3\n\n"
        "步骤2 - 分析：基于这些相关新闻的事实和解决方案，从$perspective 的角度识别出共同的原则或模式。\n"
        "## 共同的原则或模式：共同的原则或模式\n\n"
        "基于这些原则或模式，让我们分析当前的新闻。\n"
        "### 推理（从$perspective 的角度）：推理\n"
        "### 这则新闻是假的还是真的？是真是假\n"
    )
        }
        return analogical_reasoning_dict[self.source_language]
        
    def _init_template_thot_with_perspective(self):
        thot_dict = {
         "english":Template(
        "You are an expert in fake news detection analyzing news from the perspective of $perspective. Here is a news item: $news_item\n"
        "Walk me through this news item in manageable parts step by step, summarizing and analyzing as we go.\n\n"
        "## Segment 1:\n"
        "### Description (from the perspective of $perspective): Describe Segment 1\n"
        "### Analysis (from the perspective of $perspective): Analyze Segment 1\n\n"
        "## Segment 2:\n"
        "### Description (from the perspective of $perspective): Describe Segment 2\n"
        "### Analysis (from the perspective of $perspective): Analyze Segment 2\n\n"
        "## Segment 3:\n"
        "### Description (from the perspective of $perspective): Describe Segment 3\n"
        "### Analysis (from the perspective of $perspective): Analyze Segment 3\n\n"
        "... (Continue segmenting and analyzing as needed) ...\n\n"
        "Based on the analysis of the above segments, let's integrate these findings to determine the overall conclusion.\n"
        "## REASONING (from the perspective of $perspective): Reasoning\n"
        "## IS THIS NEWS FAKE OR TRUE? True or Fake"
    ),
        "german":Template(
        "Sie sind ein Experte für die Erkennung von Fake News und analysieren Nachrichten aus der Perspektive von $perspective. Hier ist ein Nachrichtenartikel: $news_item\n"
        "Führen Sie mich Schritt für Schritt durch diesen Nachrichtenartikel, indem Sie ihn in überschaubare Teile zerlegen, zusammenfassen und analysieren, während wir fortfahren.\n\n"
        "## Segment 1:\n"
        "### Beschreibung (aus der Perspektive von $perspective): Beschreiben Sie Segment 1\n"
        "### Analyse (aus der Perspektive von $perspective): Analysieren Sie Segment 1\n\n"
        "## Segment 2:\n"
        "### Beschreibung (aus der Perspektive von $perspective): Beschreiben Sie Segment 2\n"
        "### Analyse (aus der Perspektive von $perspective): Analysieren Sie Segment 2\n\n"
        "## Segment 3:\n"
        "### Beschreibung (aus der Perspektive von $perspective): Beschreiben Sie Segment 3\n"
        "### Analyse (aus der Perspektive von $perspective): Analysieren Sie Segment 3\n\n"
        "... (Setzen Sie die Segmentierung und Analyse nach Bedarf fort) ...\n\n"
        "Basierend auf der Analyse der obigen Segmente, lassen Sie uns diese Erkenntnisse integrieren, um die endgültige Schlussfolgerung zu bestimmen.\n"
        "## BEGRÜNDUNG (aus der Perspektive von $perspective): Begründung\n"
        "## IST DIESE NACHRICHT FALSCH ODER WAHR? Falsch oder Wahr"
    ),
         "bengali":Template(
        "আপনি একজন ভুয়া খবর সনাক্তকরণ বিশেষজ্ঞ এবং $perspective এর দৃষ্টিকোণ থেকে খবর বিশ্লেষণ করছেন। এখানে একটি সংবাদ আইটেম রয়েছে: $news_item\n"
        "আমাকে এই সংবাদ আইটেমটি ধাপে ধাপে পরিচালনাযোগ্য অংশে নিয়ে যান, সারাংশ তৈরি করুন এবং বিশ্লেষণ করুন যখন আমরা এগিয়ে চলি।\n\n"
        "## সেগমেন্ট ১:\n"
        "### বর্ণনা ( $perspective এর দৃষ্টিকোণ থেকে): সেগমেন্ট ১ বর্ণনা করুন\n"
        "### বিশ্লেষণ ( $perspective এর দৃষ্টিকোণ থেকে): সেগমেন্ট ১ বিশ্লেষণ করুন\n\n"
        "## সেগমেন্ট ২:\n"
        "### বর্ণনা ( $perspective এর দৃষ্টিকোণ থেকে): সেগমেন্ট ২ বর্ণনা করুন\n"
        "### বিশ্লেষণ ( $perspective এর দৃষ্টিকোণ থেকে): সেগমেন্ট ২ বিশ্লেষণ করুন\n\n"
        "## সেগমেন্ট ৩:\n"
        "### বর্ণনা ( $perspective এর দৃষ্টিকোণ থেকে): সেগমেন্ট ৩ বর্ণনা করুন\n"
        "### বিশ্লেষণ ( $perspective এর দৃষ্টিকোণ থেকে): সেগমেন্ট ৩ বিশ্লেষণ করুন\n\n"
        "... (প্রয়োজন মতো সেগমেন্টিং এবং বিশ্লেষণ চালিয়ে যান) ...\n\n"
        "উপরের সেগমেন্টগুলির বিশ্লেষণের ভিত্তিতে, আসুন এই ফলাফলগুলি একত্রিত করি এবং চূড়ান্ত সিদ্ধান্ত নির্ধারণ করি।\n"
        "## যুক্তি ( $perspective এর দৃষ্টিকোণ থেকে): যুক্তি\n"
        "## এই সংবাদটি কি ভুয়া না সত্য? ভুয়া বা সত্য"
    ),
         "arabic":Template(
             "أنت خبير في اكتشاف الأخبار المزيفة وتحليل الأخبار من منظور $perspective. إليك خبر: $news_item\n"
            "اشرح لي هذا الخبر في أجزاء يمكن إدارتها خطوة بخطوة، مع تلخيصه وتحليله أثناء تقدمنا.\n\n"
            "## الجزء 1:\n"
            "### الوصف (من منظور $perspective): وصف الجزء 1\n"
            "### التحليل (من منظور $perspective): تحليل الجزء 1\n\n"
            "## الجزء 2:\n"
            "### الوصف (من منظور $perspective): وصف الجزء 2\n"
            "### التحليل (من منظور $perspective): تحليل الجزء 2\n\n"
            "### الجزء 3:\n"
            "### الوصف (من منظور $perspective): وصف الجزء 3\n"
            "### التحليل (من منظور المنظور): تحليل الجزء 3\n\n"
            "... (استمر في التجزئة والتحليل حسب الحاجة) ...\n\n"
            "استنادًا إلى تحليل الأجزاء المذكورة أعلاه، دعنا ندمج هذه النتائج لتحديد الاستنتاج العام.\n"
            "## الاستدلال (من منظور المنظور): الاستدلال\n"
            "## هل هذه الأخبار مزيفة أم حقيقية؟ حقيقية أم مزيفة"
             ),
            "chinese":Template(
                "您是假新闻检测专家，能够从 $perspective 的角度分析新闻。这里有一条新闻：$news_item\n"
                "请逐步向我介绍这条新闻，分成几个易于管理的部分，并在介绍过程中进行总结和分析。\n\n"
                "## 片段 1:\n"
                "### 描述（从 $perspective 的角度）：描述片段 1\n"
                "### 分析（从 $perspective 的角度）：分析片段 1\n\n"
                "## 片段 2:\n"
                "### 描述（从 $perspective 的角度）：描述片段 2\n"
                "### 分析（从 $perspective 的角度）：分析片段 2\n\n"
                "## 片段 3:\n"
                "### 描述（从 $perspective 的角度）：描述片段 3\n"
                "### 分析（从 $perspective 的角度）：分析片段3\n\n"
                "...（根据需要继续细分和分析）...\n\n"
                "基于以上细分的分析，让我们整合这些发现以确定总体结论。\n"
                "## 推理（从$perspective 的角度）：推理\n"
                "## 这则新闻是假的还是真的？是真的还是假的"
            )
        }  
        return thot_dict[self.source_language]
