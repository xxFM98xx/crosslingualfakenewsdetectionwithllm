
---

### 3. **README.md für Rationale Generierung und Vorverarbeitung**

```markdown
# Rationale Generierung und Vorverarbeitung

Dieses Verzeichnis enthält Code und Skripte zur Generierung von Rationales mittels LLMs zum Vergleichen von quellsprachen Rationales gegenüber von englischen Rationales(RQ1), zur Vorverarbeitung von Datensätzen und zur Analyse des Einflusses der Sprachverteilung auf die Modellleistung (RQ2) im Rahmen der Masterarbeit **"Cross-language Fake News Detection using Large Language Models"**.

## Inhalt

### Klassen

- **PreprocessingTranslation.py**: Vorverarbeitet Datensätze und übersetzt nicht-englische Datensätze ins Englische mittels der Google Translator API. Ausführen mit `preprocess_and_translate`.

- **PreTranslationsModule.py**: Klasse zum Vorübersetzen der Datensätze mittels eines LLM. Sollte über `Pipeline.py` aufgerufen werden.

- **RationaleModule.py**: Enthält die Klasse `RationaleGenerator`, um Rationales mittels eines LLM zu generieren. Sollte über `Pipeline.py` aufgerufen werden.

- **Pipeline.py**: Orchestriert das Vorübersetzen und die Rationale-Generierung mittels der Perspektiven "Linguistic Style" und "Common Sense". Verwenden Sie `generate_rationales_without_translation` für englische Datensätze und `pretranslate_generate_rationale` für nicht-englische.

- **InformationExtractor.py**: Extrahiert Labels mittels Majority Threshold Voting, entfernt Labels aus den Rationales und extrahiert Übersetzungen aus den LLM-Antworten. Verwenden Sie `extract_all_translations` und `extract_all_rationales_labels`.

- **CorrelationAnalysisLLM.py**: Berechnet Korrelationen sprachenübergreifend sowie sprachspezifisch, zur Beantwortung der Forschungsfrage 2 (RQ2).

### Utils

- **data_prepper_script.py**: Bereitet Daten für das Modelltraining vor, indem es Datensätze aus den extrahierten Rationales und Fake News Datensätzen erstellt.

- **evaluation_script.py**: Bewertet generierte Rationales und LLM-Leistungen und stellt Funktionen zum Darstellen der Performances der Klassifikatoren Leistungen bereit.

- **extraction_script.py**: Extrahiert Rationales und Labels aus den LLM-Ausgaben.

### Skripte

- **run_falcon_7b_arg_chinese_gpu_slurm.sh**: Generiert Rationales mittels des Falcon 7B LLM auf dem ARG Chinese Datensatz.

## Anleitung zur Generierung von Rationales und Vorverarbeitung

### Voraussetzungen

- Python 3.10.5
- Installation der erforderlichen Pakete mit `requirements.txt`.
- Zugangsdaten für die Google Translator API.
- Hugging Face Token (in `main.py` setzen).

### Schritte

#### 1. Datensätze vorverarbeiten und übersetzen

### 1. Datensätze herunterladen

Links:
- ARG Datasets(Englisch und Chinesisch) (müssen angefragt werden): [Anfrageformular](https://forms.office.com/pages/responsepage.aspx?    id=DQSIkWdsW0yxEjajBLZtrQAAAAAAAAAAAAO__QiMr41UQlhTMUVHTzFLVEowWDhCODgwUjZZOTVOMi4u&route=shorturl)
- BanFakeNews: [Kaggle](https://www.kaggle.com/datasets/cryptexcode/banfakenews)
- FANG-COVID: [GitHub](https://github.com/justusmattern/fang-covid)
- AFND: [Kaggle](https://www.kaggle.com/datasets/murtadhayaseen/arabic-fake-news-dataset-afnd/data)

### 2. Datensätze speichern

Die Datensätze müssen jeweils einzeln im Ordner `cross-lingual-fake-news-detection-with-llm\Dataset\InitialDataset` gespeichert werden.

### 3. Erstelle das venv und installiere die Pakete in `requirements.txt`

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r [requirements.txt](http://_vscodecontentref_/0)

### 4. Ausführen von `PreprocessingTranslator.py`

Führen Sie `PreprocessingTranslator.py` aus, um die Vorverarbeitung wie in der Thesis beschrieben durchzuführen. Des Weiteren werden noch Zugangsdaten für die Google Translator API benötigt, die besorgt werden müssen, um die nicht-englischen Datensätze ins Englische zu übersetzen.
```

### 5. Rationale Generation

Es muss ein Huggingface Token in `main.py` gesetzt werden. In `scripts` kann ein `.sh`-File gefunden werden, welches für das LLM Falcon für den chinesischen Datensatz die Rationales generiert. Dieses muss für jedes LLM aus der Liste `[tiiuae/falcon-7b-instruct, FreedomIntelligence/phoenix-inst-chat-7b, DAMO-NLP-MT/polylm-chat-13b, Qwen/Qwen2-7B-Instruct, google/gemma-1.1-7b-it, meta-llama/Meta-Llama-3.1-8B-Instruct, meta-llama/Llama-2-7b-chat-hf, SeaLLMs/SeaLLMs-v3-7B-Chat]` in Kombination für jeden Datensatz aus der Liste `[ARGENGLISH, ARG-CHINESE, BanFakeNews, FANG-COVID, AFND]` erstellt und ausgeführt werden, um das Experiment nachzubauen.

### 6. Extraktion der Rationales

Um die Rationales aus den Skripten zu extrahieren, kann die `InformationExtraktor`-Klasse mit dem `extraction_script.py` genutzt werden, welches die in `cross-lingual-fake-news-detection-with-llm\Dataset\ProcessedDataset\` gespeicherten (kann im Script geändert werden) in das Unterverzeichnis `extracted` (wird automatisch erstellt) extrahiert, sodass ein finales Label bestimmt wird und Labels aus den Rationales extrahiert werden.

### 6.5 Evaluation

Um die durch das LLM generierten Rationales zu evaluieren, kann die `Evaluation`-Klasse genutzt werden. Diese enthält diverse Funktionen, um datensatzübergreifend in einem DataFrame (`cross_ds_evaluation()`), per Datensatz (`_all_llm_evaluation_into_dfs`) sowie auf Level eines einzelnen LLMs (`evaluate_llm()`) die LLMs bzw. die generierten Rationales zu evaluieren.

#### 6.5.1 evaluate_llm()

Um Rationales eines LLMs auf einem Datensatz zu evaluieren, kann `Evaluator.evaluate_llm()` aufgerufen werden, wobei das DataFrame, welches die Rationales enthält, in der `predicted`-Spalte auf 0 und 1 gefiltert werden muss, bevor die Funktion aufgerufen werden kann. Es speichert `llm_report.json` für jede Datensatzvariante in einem Unterverzeichnis im spezifizierten Verzeichnis `report_path`.

#### 6.5.2 _all_llm_evaluation_into_dfs()

Um die durch `evaluate_llm` in Unterverzeichnissen in JSON gespeicherten `llm_report.json` in DataFrames bzw. Tabellen, wie in der Thesis zu sehen, je Datensatz zu übertragen, kann `Evaluator._all_llm_evaluation_into_dfs()` genutzt werden. Dafür muss `report_dir` angegeben werden, welches alle `llm_report.json` datensatzübergreifend enthalten sollte.

### 7. Erstellen der Datensätze für nachfolgendes Training der Modelle

Um Datensätze aus den vorher generierten und nachfolgend extrahierten DataFrames, welche im Unterverzeichnis `extracted` gespeichert sind, zu erstellen, kann das `Data_Prepper`-Script genutzt werden. Das nimmt die DataFrames, die in `extracted` gespeichert sind, und erstellt ein neues Unterverzeichnis `prepped` auf Ebene des Unterverzeichnisses `extracted` und erstellt dort je LLM- und Datensatz-Kombination ein Unterverzeichnis, z.B. `_falcon_7b_arg_chinese`. Dieses Unterverzeichnis `falcon_7b_arg_chinese` enthält Unterunterverzeichnisse für jede Datensatzvariante `source`, `google`, `llm`, welche dann `train.json`, `test.json` und `val.json` enthalten.

### RQ2

Um den Einfluss der Sprachverteilung in den Trainingsdaten auf die Performance zu untersuchen:

1. Zuerst müssen die LLM_Performances evaluiert werden durch `Evaluator.evaluate_llm()`, falls nicht schon vorher geschehen, und in `cross-lingual-fake-news-detection-with-llm\Reports` gespeichert werden.
2. Aufrufen von `evaluation_script.py` durch Aufrufen der Funktionen `all_arg_llm_cross_ds_(LLM_REPORT_PATH)`, sodass ein datensatzübergreifender LLM Performances DataFrame erstellt wird, wie in der Thesis zu sehen.
3. Aufrufen von `evaluation_script.py` `transform_llm_into_la_info_supported(pd.read_csv(LLM_CROSS_DS_PATH))` zum Erstellen der DataFrames `data_df`, `binary_data_df`, welche die LLM Performances sowie die Sprachverteilungen in Form von unterstützten Sprachen bzw. numerische Verteilungen enthält.
4. Ausführen von `CorrelationAnalysisLLM.py`, um Korrelationen in Form von DataFrames zu erhalten.



