"""
Description:
This class is used to evaluate the performance of the LLMs and other classifiers. It can evaluate the performance of a LLM model, all classifier models, and cross-dataset evaluation.
The class can also aggregate the metrics for grouping and extract the classifier type and source type from the model name.

"""

from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score, classification_report
import pandas as pd
import json
import numpy as np
from typing import Union
import time
import os
from collections import defaultdict
from utils import read_files_into_list_of_dicts, 

class Evaluator:
    """
    Class to evaluate the performances of a LLMs and other classifiers
    
    Note: 
        1. (performance of trained classifiers is measured after training directly but here it is aggregated to compare dataset variants to evaluate the Approach)
        2. One needs to filter other labels from the data before passing it to the evaluator as it can only accept real and fake labels(0,1)
    
    
    Args:
    ground_truth_column_name: list of strings
        The column name of the ground truth labels
    predicted_column_names: list of strings
        The column names of the predicted labels
    label_mapping: dict
        Mapping of the labels to their corresponding classes
    """
    def __init__(self, ground_truth_column_name:str, predicted_column_names:list[str], label_mapping:dict={0:'Real', 1:'Fake'}):
        """
        Args:
            ground_truth_column_name (str): 
            predicted_column_names (list[str]): 
            label_mapping (_type_, optional):  Defaults to {0:'Real', 1:'Fake'}.
        """
        self.ground_truth_column_name = ground_truth_column_name
        self.predicted_column_names = predicted_column_names
        self.label_mapping = label_mapping


    def evaluate_llm(self, model_name:str, data:Union[pd.DataFrame,list[dict]], report_path:str):
        """
        Evaluate the performance of a LLM model on a dataset.
        
        Args:
            model_name (str): LLM model name
            data (Union[pd.DataFrame,list[dict]]): Data to evaluate 
            report_path (str): Path to save the report 

        Raises:
            ValueError: If the ground truth column is not found in the data
            ValueError: If the predicted column is not found in the data
            ValueError: If the data is not a list of dictionaries or a pandas DataFrame
        
        Additional Information:
        predicted_column_names in Data needs to filtered to only contain real and fake labels(0,1) before passing it to the evaluate_llm function.
         
        Example Report one(llm) out three(llm, google, source) reports:
        {
    "td_pred": {
        "Real": {
            "precision": 0.8936170212765957,
            "recall": 0.5230769230769231,
            "f1-score": 0.6598890942698706,
            "support": 4095.0
        },
        "Fake": {
            "precision": 0.34943371085942704,
            "recall": 0.8044478527607362,
            "f1-score": 0.4872271249419415,
            "support": 1304.0
        },
        "accuracy": 0.5910353769216522,
        "macro avg": {
            "precision": 0.6215253660680113,
            "recall": 0.6637623879188297,
            "f1-score": 0.573558109605906,
            "support": 5399.0
        },
        "weighted avg": {
            "precision": 0.7621824895514636,
            "recall": 0.5910353769216522,
            "f1-score": 0.6181867034560866,
            "support": 5399.0
        },
        "model_name": "falcon_7b_arg_llm",
        "time": 1731436996.1417854
    },
    "cs_pred": {
        "Real": {
            "precision": 0.8930084745762712,
            "recall": 0.4117216117216117,
            "f1-score": 0.5635968577636637,
            "support": 4095.0
        },
        "Fake": {
            "precision": 0.31387069211050983,
            "recall": 0.8450920245398773,
            "f1-score": 0.457736240913811,
            "support": 1304.0
        },
        "accuracy": 0.5163919244304501,
        "macro avg": {
            "precision": 0.6034395833433905,
            "recall": 0.6284068181307445,
            "f1-score": 0.5106665493387373,
            "support": 5399.0
        },
        "weighted avg": {
            "precision": 0.7531315217451261,
            "recall": 0.5163919244304501,
            "f1-score": 0.5380287443403987,
            "support": 5399.0
        },
        "model_name": "falcon_7b_arg_llm",
        "time": 1731436996.1417854
    }
}
        """
        # Classification Report
        if type(data) == list:
            data = self._list_json_to_dataframe(data)
        elif type(data) == pd.DataFrame:
            pass
        else:
            raise ValueError('Data must be either a list of dictionaries or a pandas DataFrame')
        
        if self.ground_truth_column_name not in data.columns:
            raise ValueError(f'Ground truth column "{self.ground_truth_column_name}" not found in the data.')

        ground_truth = data[self.ground_truth_column_name]
        
        unique_labels = np.unique(ground_truth)
        target_names = [self.label_mapping[label] for label in unique_labels]
        timestamp = time.time()
        
        all_reports = {}
        for column in self.predicted_column_names:
            if column not in data.columns:
                raise ValueError(f'Predicted column "{column}" not found in the data.')

            prediction = data[column]
            report_dict = classification_report(ground_truth, prediction, target_names=target_names, output_dict=True)
            
            report_dict['model_name'] = model_name
            report_dict['time'] = timestamp
            
            # Store the report in the dictionary under the column name
            all_reports[column] = report_dict
        
        # Save the entire reports dictionary to a JSON file
        self._save_report(all_reports, report_path)
    

    def all_classifier_evaluation_into_several_csv(self, data:Union[pd.DataFrame,list[dict]], report_path:str):
        """
        Evaluate the performance of the classifier models and save the reports in separate CSV files
        
        Args:
            data (Union[pd.DataFrame,list[dict]]): Data to evaluate 
            report_path (str): Path to save the report
        
        Raises:
            ValueError: If the data is not a list of dictionaries or a pandas DataFrame
        
        
        Returns:
            pd.DataFrame: A pandas DataFrame containing the cleaned and sorted data. Report 
        
        Additional Information:
            Save the report in the scheme: "{report_path}/all_classifier_reports_{dataset_type}_cleaned.csv"
            Where does the data come from?-> From the ARG model and other Classifier models
            
            One can collect the logs from the ARG model and other classifier models and then pass it to this function
            by saving all logs in a dictionary and calling read_files_into_list_of_dicts function (see utils/__init__.py)
            
            Example:
            data = read_files_into_list_of_dicts("path/to/logs")
        """
        dataset_reports = {
            'fang': [],
            'asdn': [],
            'arg_chinese': [],
            'arg': [],
            'ban': []
            }
        classifier_name = {
            'ARG': [],
            'RoBERTa_MLP': [],
            'RoBERTa_CNN_MLP': [],
        }
        source_type = {
            'source': [],
            'llm': [],
            'google': [],
        }
        metric_columns = [
            "f1_real", "f1_fake", "acc", "metric"
        ]
        
        for model_evaluated in data:
            for llm_model_name, evaluation in model_evaluated.items():
                dataset_type = self._extract_dataset_type(llm_model_name)
                print(f"dataset_type: {dataset_type}")
                if dataset_type == 'asnd':
                    dataset_type = 'asdn'
                #Substract the dataset type from the model name
                llm_model_name_extracted = llm_model_name.replace(f"_{dataset_type}", "")

                classifier_name = llm_model_name_extracted.split("_")[-1] if not "rationale" in llm_model_name_extracted else llm_model_name_extracted.split("_")[-2]
                print(f"model_name_extracted: {llm_model_name_extracted}")
                

                # Add the model name to the evaluation dictionary
                evaluation['model_name'] = llm_model_name_extracted

                # Add the evaluation to the corresponding dataset type dict list
                dataset_reports[dataset_type].append(evaluation)
        print(f"dataset_reports: {dataset_reports}")

        for dataset_type, reports in dataset_reports.items():
            if reports:
                df = self._list_json_to_dataframe(reports)
                new_rows = []

                # Iterate over the models and perform calculations
                for model_name in df["model_name"].unique():
                    if "source" in model_name:
                        # Identify the baseline model
                        baseline_model = model_name
                        corresponding_llm_model = model_name.replace("source", "llm")
                        corresponding_google_model = model_name.replace("source", "google")

                        # Filter the data for the relevant models
                        baseline_data = df[df["model_name"] == baseline_model]
                        llm_data = df[df["model_name"] == corresponding_llm_model]
                        google_data = df[df["model_name"] == corresponding_google_model]
                        print(f"baseline_data: {baseline_data}")
                        print(f"llm_data: {llm_data}")
                        print(f"google_data: {google_data}")

                        # Check if relevant data exists
                        if not baseline_data.empty and not llm_data.empty:
                            # Calculate relative improvement for LLM model
                            new_row = baseline_data.iloc[0].copy()
                            new_row["model_name"] = f"{corresponding_llm_model}_relative_improvement"
                            for metric in metric_columns:
                                if metric in baseline_data.columns and metric in llm_data.columns:
                                    baseline_value = baseline_data.iloc[0][metric]
                                    llm_value = llm_data.iloc[0][metric]
                                    if baseline_value != 0:
                                        relative_improvement = ((llm_value - baseline_value) / baseline_value) * 100
                                    else:
                                        relative_improvement = 0
                                    new_row[metric] = relative_improvement
                            new_rows.append(new_row)


                        if not google_data.empty and not llm_data.empty:
                            # Calculate difference between Google and LLM model
                            new_row = google_data.iloc[0].copy()
                            new_row["model_name"] = f"{corresponding_llm_model}_google_difference"
                            for metric in metric_columns:
                                if metric in google_data.columns and metric in llm_data.columns:
                                    google_value = google_data.iloc[0][metric]
                                    llm_value = llm_data.iloc[0][metric]
                                    difference = llm_value - google_value
                                    new_row[metric] = difference
                            new_rows.append(new_row)
                
                # Add new rows to the DataFrame at the correct positions
                for new_row in new_rows:
                    base_model_name = new_row["model_name"].replace("_relative_improvement", "").replace("_google_difference", "")
                    insertion_index = df[df["model_name"] == base_model_name].index.max() + 1
                    df = pd.concat([df.iloc[:insertion_index], pd.DataFrame([new_row]), df.iloc[insertion_index:]]).reset_index(drop=True)

                # Sort the data by model names to ensure similar models are grouped together
                df = df.sort_values(by="model_name").reset_index(drop=True)
                
                # Calculate the standard deviation for each category
                for category in ["relative_improvement", "google_difference", "performance"]:
                    if category == "performance":
                        category_rows = df[~df["model_name"].str.contains("relative_improvement|google_difference")]
                    else:
                        category_rows = df[df["model_name"].str.contains(category)]
                    
                    if not category_rows.empty:
                        std_devs = category_rows[metric_columns].std()
                        for index, row in category_rows.iterrows():
                            for metric in metric_columns:
                                if metric in row:
                                    value = row[metric]
                                    std_dev = std_devs[metric]
                                    print(f"before dtypes: {df.dtypes}")
                                    df[metric] = df[metric].astype(object)
                                    print(f"after dtypes: {df.dtypes}")
                                    if category == "relative_improvement":
                                        # Format the relative improvement as a percentage
                                        df.at[index, metric] = f"{round(value, 2)}% ± {round(std_dev, 2)}%"
                                    else:
                                        df.at[index, metric] = f"{round(value, 2)} ± {round(std_dev, 2)}"

                # Save the cleaned and sorted DataFrame to a CSV file
                df = df[['model_name','acc', 'metric','f1_real','f1_fake']]
                df.to_csv(f"{report_path}/all_classifier_reports_{dataset_type}_cleaned.csv", index=False)


    def _list_json_to_dataframe(self, data:list[dict]):
        """
        Convert a list of dictionaries to a pandas DataFrame
        
        Args:
            data (list[dict]): A list of dictionaries
        
        Returns:
            pd.DataFrame: A pandas DataFrame
        """
        return pd.DataFrame(data)
    

    def _save_report(self, report:dict, report_name:str):
        """
        Save the report to a JSON file
        
        Args:
            report (dict): The report to save
            report_name (str): The name of the report
        """
        with open(f'{report_name}/llm_report.json', 'w') as f:
            json.dump(report, f, indent=4)
    

    def cross_ds_evaluation(self, list_of_reports: dict, report_path: str):
        """
        Evaluate the performance of the LLM model across different datasets by aggregating the weighted F1 metrics.
        Save the report in the scheme: report_path/cross_ds_evaluation.csv

        Args:
            list_of_reports (dict): Dictionary containing report data for each dataset.
            report_path (str): The path where the cross-dataset evaluation report will be saved.
        
        Additional Information:
        Can be comfortably called in evaluation_script.py to evaluate the performance of the LLM model across different datasets.
        
        """
        # Define the metrics of interest
        key_metrices = ['td_pred_weighted_f1', 'cs_pred_weighted_f1']
        rows = []

        # Iterate through each report and extract model name, dataset type, and source type
        for filename, report_df in list_of_reports.items():
            # Extract model names, dataset types, and source types
            model_name = report_df['model_name'].str.split('_').str[0]
            source_type = report_df['model_name'].str.split('_').str[-1]
            dataset_type = report_df['model_name'].apply(lambda x: self._extract_dataset_type(x))
            # Manuelle Anpassung der Quelltypen falls notwendig
            source_type = source_type.apply(lambda x: report_df['model_name'].str.split('_').str[-2] if x in ['relative_improvement', 'google_difference'] else x)

            # Add metrics to the `rows` list
            for model, dataset, source, td_f1, cs_f1 in zip(model_name, dataset_type, source_type,
                                                            report_df['td_pred_weighted_f1'], report_df['cs_pred_weighted_f1']):
                rows.append({
                    'modelname_source': f"{model}_{source}",
                    f"{dataset}_td_pred_weighted_f1": td_f1,
                    f"{dataset}_cs_pred_weighted_f1": cs_f1
                })

        # Create a DataFrame from the `rows` list
        df = pd.DataFrame(rows)

        # Group by `modelname_source` and aggregate the metrics	
        df = df.groupby('modelname_source').first().reset_index()
        df.fillna('-', inplace=True)

        # Show the resulting DataFrame
        print(df)

        df.to_csv(f"{report_path}/cross_ds_evaluation.csv", index=False)



    def _aggregate_metrics(self, series):
        """
        Aggregate the metrics for grouping.
        

        Args:
            series (pd.Series): Series of values to be aggregated.

        Returns:
            The aggregated value.        
        """
        if series.nunique() == 1:
            return series.iloc[0]
        return series.mean() if pd.api.types.is_numeric_dtype(series) else ', '.join(map(str, series.unique()))

    def extract_classifier(self,model_name:str):
        """
        Extract the classifier type from the model name

        Args:
            model_name (str): The model name string

        Returns:
            str: The classifier type extracted from the model name
        """
        classifier_types = ['ARG', 'RoBERTa_MLP', 'RoBERTa_CNN_MLP']
        for classifier_type in classifier_types:
            if classifier_type in model_name:
                return classifier_type
        return 'unknown'
    
    def extract_source_type(self,model_name:str):
        """
        Extract the source type from the model name

        Args:
            model_name (str): The model name string

        Returns:
            str: The source type extracted from the model name
        """
        source_types = ['source', 'llm', 'google']
        for source_type in source_types:
            if source_type in model_name:
                return source_type
        return 'unknown'
                        
                        
    def _all_llm_evaluation_into_dfs(self, report_dir: str, report_name: str = None):
        """
        Write all LLM Performance evaluations into separate DataFrames based on dataset type and save them as CSV files
        like the scheme: report_dir/all_llm_reports_{dataset_type}_cleaned.csv        
        
        Args:
            report_dir (str): Where the reports going to be used are stored
            report_name (str): The name of the single report which is to be generated
        """
        dataset_reports = {
            'fang': [],
            'asdn': [],
            'arg_chinese': [],
            'arg': [],
            'ban': []
        }

        for root, dirs, files in os.walk(report_dir):
            for file in files:
                if file == "llm_report.json" if report_name is None else report_name:
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        report = json.load(f)
                        flattened_reports = self._extract_metrics_from_report(report)
                        for entry in flattened_reports:
                            dataset_type = self._extract_dataset_type(entry.get('model_name', 'unknown_model'))
                            if dataset_type in dataset_reports:
                                dataset_reports[dataset_type].append(entry)

        if all(len(reports) == 0 for reports in dataset_reports.values()):
            raise ValueError('No LLM reports found in the directory.')

        for dataset_type, reports in dataset_reports.items():
            if reports:
                df = pd.DataFrame(reports)
                metric_columns = [
                    "td_pred_real_f1-score", "td_pred_fake_f1-score", "td_pred_acc", "td_pred_weighted_f1",
                    "cs_pred_real_f1-score", "cs_pred_fake_f1-score", "cs_pred_acc", "cs_pred_weighted_f1"
                ]
                new_rows = []
                
                if dataset_type != 'arg':
                    # Iterating over the models and performing calculations
                    for model_name in df["model_name"].unique():
                        if "source" in model_name:
                            # Identify the baseline model
                            baseline_model = model_name
                            corresponding_llm_model = model_name.replace("source", "llm")
                            corresponding_google_model = model_name.replace("source", "google")

                            # Filter the data for the relevant models
                            baseline_data = df[df["model_name"] == baseline_model]
                            llm_data = df[df["model_name"] == corresponding_llm_model]
                            google_data = df[df["model_name"] == corresponding_google_model]

                            # Check if relevant data exists
                            if not baseline_data.empty and not llm_data.empty:
                                # Calculate relative improvement for LLM model
                                new_row = baseline_data.iloc[0].copy()
                                new_row["model_name"] = f"{corresponding_llm_model}_relative_improvement"
                                for metric in metric_columns:
                                    if metric in baseline_data.columns and metric in llm_data.columns:
                                        baseline_value = baseline_data.iloc[0][metric]
                                        llm_value = llm_data.iloc[0][metric]
                                        if baseline_value != 0:
                                            relative_improvement = ((llm_value - baseline_value) / baseline_value) * 100
                                        else:
                                            relative_improvement = 0
                                        new_row[metric] = relative_improvement
                                new_rows.append(new_row)

                            if not google_data.empty and not llm_data.empty:
                                # Calculate difference between Google and LLM model
                                new_row = google_data.iloc[0].copy()
                                new_row["model_name"] = f"{corresponding_llm_model}_google_difference"
                                for metric in metric_columns:
                                    if metric in google_data.columns and metric in llm_data.columns:
                                        google_value = google_data.iloc[0][metric]
                                        llm_value = llm_data.iloc[0][metric]
                                        difference = llm_value - google_value
                                        new_row[metric] = difference
                                new_rows.append(new_row)

                    # Add new rows to the DataFrame at the correct positions
                    for new_row in new_rows:
                        base_model_name = new_row["model_name"].replace("_relative_improvement", "").replace("_google_difference", "")
                        insertion_index = df[df["model_name"] == base_model_name].index.max() + 1
                        df = pd.concat([df.iloc[:insertion_index], pd.DataFrame([new_row]), df.iloc[insertion_index:]]).reset_index(drop=True)

                # Sort the data by model names to ensure similar models are grouped together
                df = df.sort_values(by="model_name").reset_index(drop=True)


                # Calculate the standard deviation for each category
                for category in ["relative_improvement", "google_difference", "performance"]:
                    if category == "performance":
                        category_rows = df[~df["model_name"].str.contains("relative_improvement|google_difference")]
                    else:
                        category_rows = df[df["model_name"].str.contains(category)]
                    
                    if not category_rows.empty:
                        std_devs = category_rows[metric_columns].std()
                        for index, row in category_rows.iterrows():
                            for metric in metric_columns:
                                if metric in row:
                                    value = row[metric]
                                    std_dev = std_devs[metric]
                                    if category == "relative_improvement":
                                        # Format the relative improvement as a percentage
                                        df.at[index, metric] = f"{round(value, 2)}% ± {round(std_dev, 2)}%"
                                    else:
                                        df.at[index, metric] = f"{round(value, 2)} ± {round(std_dev, 2)}"
                # Save the cleaned and sorted DataFrame to a CSV file
                df.to_csv(f"{report_dir}/all_llm_reports_{dataset_type}_cleaned.csv", index=False)



    def _extract_metrics_from_report(self, report: dict):
        """
        Flatten the JSON report to extract metrics into a list of dictionaries.
        
        Args:
            report (dict): The JSON report to be flattened

        Returns:
            list: A list of flattened dictionaries representing the report
        """
        flattened_data = []
        for perspective_key, metrics in report.items():
            if perspective_key not in ["model_name", "time"]:
                model_name = metrics.get("model_name", "unknown_model")
                print(f"metrics: {metrics}")
                for metric_key, metric_value in metrics.items():
                    if isinstance(metric_value, dict):
                        # Flatten the metrics like "Real", "Fake", "macro avg", "weighted avg"
                        flattened_entry = {
                            'model_name': model_name,
                        }
                        if metric_key == "weighted avg":
                            print(f"metric_key in weighted avg: {metric_key}")
                            for metric_name, metric_value in metric_value.items():
                                if metric_name == "f1-score":
                                    column_name = perspective_key + "_weighted_f1"
                                    flattened_entry[column_name] = metric_value
                                
                        if metric_key == "Real":
                            print(f"metric_key in Real: {metric_key}")
                            for metric_name, metric_value in metric_value.items():
                                if metric_name == "f1-score":
                                    column_name = perspective_key +"_real"+ "_f1-score"
                                    flattened_entry[column_name] = metric_value

                        elif metric_key == "Fake":
                            print(f"metric_key in Fake: {metric_key}")
                        
                            for metric_name, metric_value in metric_value.items():
                                if metric_name == "f1-score":
                                    column_name = perspective_key +"_fake"+ "_f1-score"
                                    flattened_entry[column_name] = metric_value
                        
                    elif metric_key == "accuracy":
                        column_name = perspective_key + "_acc"
                        flattened_entry[column_name]= metric_value

                    print(f"flattened_entry: {flattened_entry}")
                    flattened_data.append(flattened_entry)
        flattened_data = self.remove_duplicates(flattened_data)
        flattened_data = self.merge_dicts(flattened_data, 'model_name')
        print(f"flattened_data: {flattened_data}")
        return flattened_data
    
    def remove_duplicates(self, dict_list):
        """
        Remove duplicate dictionaries from a list
        
        Args:
            dict_list (list): A list of dictionaries
            
            Returns:
            list: A list of dictionaries with duplicates removed
        """

        seen = set()
        unique_dicts = []
        for d in dict_list:
            t = tuple(sorted(d.items()))
            if t not in seen:
                seen.add(t)
                unique_dicts.append(d)
        return unique_dicts
    
    def merge_dicts(self, dict_list, key):
        """
        Merge a list of dictionaries based on a key
        
        Args:
            dict_list (list): A list of dictionaries
            key (str): The key to merge the dictionaries on
            
            Returns:
            list: A list of merged dictionaries
            """
        merged_dict = defaultdict(dict)
        for d in dict_list:
            merged_dict[d[key]].update(d)
        return list(merged_dict.values())
    
    def _extract_dataset_type(self, model_name: str):
        """
        Extract the dataset type from the model name

        Args:
            model_name (str): The model name string

        Returns:
            str: The dataset type extracted from the model name
        """
        dataset_types = ['fang', 'asdn', 'arg_chinese', 'arg', 'ban', 'asnd']
        for dataset_type in dataset_types:
            if dataset_type in model_name:
                return dataset_type
        return 'unknown'

    
