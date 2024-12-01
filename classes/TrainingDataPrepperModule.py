import pandas as pd
import uuid
from sklearn.model_selection import train_test_split

class DataPrepper:
    """
    A class to prepare data for model training and evaluation.
    
    Parameters:
        df (pd.DataFrame): The dataset containing the text data.
        rationale_column_names (list[str]): List of rationale column names to be processed.
        perspectives (list[str]): List of perspectives to be used in processing.
        source_column (str): Column name containing the text data.
        **kwargs: Additional keyword arguments for configuring the data prepper (e.g., split ratios).
    """

    def __init__(self, df: pd.DataFrame, rationale_column_names: list[str], perspectives: list[str], source_column: str, **kwargs):        
        self.perspectives = perspectives
        self.source_column = source_column
        
        # Use default split data or override with provided kwargs
        self.split_data = kwargs.get("split_data", {"train": 0.8, "val": 0.1, "test": 0.1})
        
        # Generate prediction column names based on rationale columns
        self.prediction_column_names = self._generate_prediction_column_names(rationale_column_names)
        
        # Generate rationale column names based on rationale columns
        self.rationale_column_names = self._generate_rationale_column_names(rationale_column_names)
        
        dataset = df.copy()
        self.dataset = dataset.dropna(subset=self.rationale_column_names + self.prediction_column_names)
        # wenn translated im columnnamen: falls eines der Zeilen aus den Spalten null enthält, dann entferne diese Zeile
        # wenn rationale_column_names: falls eines der Zeilen aus den Spalten null enthält, dann entferne diese Zeile

        self.dataset.loc[:, 'unique_id'] = list(range(1, len(self.dataset) + 1))
        self.label_mapping = {"true": 0, "fake": 1,"real": 0,"other": 2}
        self.label_column = "label" if "label" in self.dataset.columns else ("source_label" if "source_label" in self.dataset.columns  else raise ValueError("No supported label column found in the dataset."))       
    def _generate_prediction_column_names(self,rationale_column_names:list[str]) -> list[str]:
        """
        Generate prediction column names based on the provided rationale column names.
        
        Parameters:
            rationale_column_names (list[str]): List of rationale column names.
        
        Returns:
            list[str]: List of generated prediction column names.
        """
        return [f"{column}_extracted_label" for column in rationale_column_names]
    
    def _generate_rationale_column_names(self,rationale_column_names:list[str]) -> list[str]:
        """
        Generate rationale column names based on the rationale column names.
        
        Parameters:
            rationale_column_names (list[str]): List of rationale column names.
        
        Returns:
            list[str]: List of generated rationale column names.
        """
        
        return [f"{column}_extracted_rationale" for column in rationale_column_names]
        

    def _validate_split(self, split: str):
        """
        Validate the provided data split (train, val, test).
        
        Raises:
            ValueError: If the split is not one of 'train', 'val', or 'test'.
        """
        allowed_splits = ["train", "val", "test"]
        if split not in allowed_splits:
            raise ValueError(f"Invalid split value: {split}. Allowed values are: {allowed_splits}")
    
    def _process_row(self, row, split: str, label: int) -> dict:
        """
        Process a single row of data and return it in a transformed format.
        
        Parameters:
            row (pd.Series): A single row of the dataset.
            split (str): The data split (train, val, test).
            label (int): The ground truth label for this row.
            datapoint_json_template (dict): The template for generating the JSON data structure.
        
        Returns:
            dict: The processed data row in JSON format.
        """
        datapoint_json = {}
        abbreviation_dict = {"linguistic style": "td", "commonsense": "cs", "common sense": "cs"}
        datapoint_json["content"] = row[self.source_column].replace('\n', ' ')
        datapoint_json["source_id"] = row["unique_id"]
        datapoint_json["split"] = split
        datapoint_json["label"] = label
        rationale_prediction_columns = self.rationale_column_names + self.prediction_column_names
        for column in rationale_prediction_columns:
            abbrev = None  
            for perspective in self.perspectives:
                if perspective["english"] in column.replace("_"," ") and perspective["english"] in abbreviation_dict:
                    abbrev = abbreviation_dict[perspective["english"]]
                    break
                
            if abbrev is None:
                continue
             
            if column in self.rationale_column_names:
                datapoint_json[f"{abbrev}_rationale"] = row[column].replace('\n', ' ')
            elif column in self.prediction_column_names:
                try:
                    prediction = self.label_mapping[str.lower(row[column])]
                except KeyError:
                    prediction = self.label_mapping["other"]
                datapoint_json[f"{abbrev}_pred"] =  prediction
                datapoint_json[f"{abbrev}_acc"] = 1 if prediction == label else 0
        
        for time_col in ['time', 'date']:
            if time_col in row:
                datapoint_json["time"] = row[time_col].replace('\n', ' ') if type(row[time_col]) == str else row[time_col]
                break

        return datapoint_json

    def _arg_transform_data(self, split: str, df: pd.DataFrame=None):
        """
        Transform the data for training and evaluation of the arg network.
        
        Parameters:
            split (str): The split type (train, val, test).
            df (pd.DataFrame): Optional dataframe to transform. If None, uses the class dataset.
        
        Returns:
            list[dict]: Transformed data in JSON format.
        """
        self._validate_split(split)

        data = df if df is not None else self.dataset

        def process_row_wrapper(row):
            label = row[self.label_column]
            if isinstance(label, str):
                label = label.strip().lower()
            if label not in self.label_mapping and not isinstance(label, int) and label not in self.label_mapping.values():
                return None
                #raise ValueError(f"Invalid label found: {label}")
            
            label = row[self.label_column] if isinstance(row[self.label_column], int) else self.label_mapping[row[self.label_column]]

            return self._process_row(row, split, label)
        
        return [result for result in data.apply(process_row_wrapper, axis=1).tolist() if result is not None]


    def shuffle_and_split_data(self, df: pd.DataFrame=None):
        """
        Shuffle and split the dataset into training, validation, and test sets.
        
        Parameters:
            df (pd.DataFrame): Optional dataset to split. If None, the class dataset is used.
        
        Returns:
            tuple: Dataframes for train, validation, and test sets.
        """
        data = df if df is not None else self.dataset
        train_ratio = self.split_data["train"]
        val_ratio = self.split_data["val"]
        test_ratio = self.split_data["test"]

        # First split: train and temp (val + test)
        train_data, temp_data = train_test_split(data, test_size=(1 - train_ratio))
        
        # Second split: val and test
        val_ratio_relative = val_ratio / (val_ratio + test_ratio)
        val_data, test_data = train_test_split(temp_data, test_size=1 - val_ratio_relative)

        return train_data, val_data, test_data

    def get_arg_datasets(self):
        """
        GO TO FUNCTION to retrieve the datasets for ARG and other classifier training and evaluation.
        Retrieve the datasets for ARG network training and evaluation.
        
        Returns:
            tuple: JSON-formatted training, validation, and test data.
        """
        train_data_df, val_data_df, test_data_df = self.shuffle_and_split_data()
        train_data_json = self._arg_transform_data("train", train_data_df)
        val_data_json = self._arg_transform_data("val", val_data_df)
        test_data_json = self._arg_transform_data("test", test_data_df)
        
        return train_data_json, val_data_json, test_data_json