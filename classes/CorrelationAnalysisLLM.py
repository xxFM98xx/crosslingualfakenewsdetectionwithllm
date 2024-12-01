# Full Python code for performing correlation


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import numpy as np

LLM_REPORT_LA_DISTRIBUTION_PATH = r"C:\Users\FilmonMesfun\Desktop\MT_FM\cross-lingual-fake-news-detection-with-llm\Reports\result_df_report_la_info_distribution.csv"
SUPPORTED_REPORT_LA_SUPPORTED_PATH = r"C:\Users\FilmonMesfun\Desktop\MT_FM\cross-lingual-fake-news-detection-with-llm\Reports\result_df_report_la_info_supported.csv"
CORRELATION_LANGUAGE_SPECIFIC_PATH = r"C:\Users\FilmonMesfun\Desktop\MT_FM\cross-lingual-fake-news-detection-with-llm\Reports\language_specific_correlations.csv"
CORELLATION_CROSS_LANGUAGE_PATH = r"C:\Users\FilmonMesfun\Desktop\MT_FM\cross-lingual-fake-news-detection-with-llm\Reports\cross_language_correlations.csv"



# Reading the data
df_distribution = pd.read_csv(LLM_REPORT_LA_DISTRIBUTION_PATH)
df_supported = pd.read_csv(SUPPORTED_REPORT_LA_SUPPORTED_PATH)

# Overview of the data
print("Dataframe structure - Distribution")
print(df_distribution.head())
print("Datastructure - Supported Languages")
print(df_supported.head())


# Replace non-numeric values like '-' with NaN
def preprocess_data(df_distribution, df_supported):
    """
    Filters out the non-numeric values and replaces them with NaN.
    
    Args:
        df_distribution: DataFrame
        df_supported: DataFrame
    
    Returns:
        df_distribution, df_supported: DataFrames
    """
    df_distribution.replace("-", np.nan, inplace=True)
    df_supported.replace("-", np.nan, inplace=True)

    # Converts all relevant columns to numeric data types
    numerical_columns = ['F1 (Google)', 'F1 (LLM)', 'F1 (Source)', 
                        'Δ F1 (LLM - Source language)', 'Δ F1 (Google - Source language)',
                        'en', 'de', 'zh', 'ar', 'bn']
    df_distribution[numerical_columns] = df_distribution[numerical_columns].apply(pd.to_numeric, errors='coerce')
    df_supported[numerical_columns] = df_supported[numerical_columns].apply(pd.to_numeric, errors='coerce')

    # Check the data after cleaning
    print("Data after cleaning - Distribution")
    print(df_distribution.head())
    print("\nData after cleaning - Supported Languages")
    print(df_supported.head())
    return df_distribution, df_supported

def calculate_correlation(data, feature_columns, target_columns, method='spearman'):
    """
    Calculate the correlation and p-values between feature and target columns.

    Args:
        data: DataFrame
        feature_columns: List of columns for the features (e.g. language distribution)
        target_columns: List of columns for the targets (e.g. F1 metrics)
        method: Correlation method ('spearman')

    Returns:
        DataFrame with correlations and p-values
    """
    correlations = []
    for feature in feature_columns:
        for target in target_columns:
            if method == 'spearman':
                corr, p_value = spearmanr(data[feature], data[target], nan_policy='omit')
                print(f"Correlation between {feature} and {target} is {corr} with p-value {p_value}")
            correlations.append({'Feature': feature, 'Target': target, 'Correlation': round(corr, 3), 'P-Value': round(p_value, 3)})
    return pd.DataFrame(correlations)


def language_specific_analysis(data, language_column, feature_columns, target_columns, method='spearman'):
    """
    Execute a language-specific analysis (correlations).
    
    Args:
        data: DataFrame
        language_column: Column with languages
        feature_columns: List of feature columns (e.g. language distribution)
        target_columns: List of target columns (e.g. F1 metrics)
        method: Correlation method ('spearman')

    Returns:
        DataFrame with language-specific correlations.
    """
    results = []

    # Get unique languages
    unique_languages = data[language_column].dropna().unique()
    


    for language in unique_languages:
        # Filter for specific language
        language_data = data[data[language_column] == language]

        # Calculate correlations for the specific language
        correlations = calculate_correlation(language_data, feature_columns, target_columns, method=method)

        # Add language to the results
        correlations['Language'] = language

        # Append results to the list
        results.append(correlations)

    # Concatenate all results into a single DataFrame
    final_results = pd.concat(results, ignore_index=True)

    return final_results


df_distribution, df_supported = preprocess_data(df_distribution, df_supported)
# Analysis: Correlations between language distribution and performance
features = ['en', 'de', 'zh', 'ar', 'bn']
targets = ['F1 (Google)', 'F1 (LLM)', 'F1 (Source)', 
           'Δ F1 (LLM - Source language)', 'Δ F1 (Google - Source language)']

correlation_distribution = calculate_correlation(df_distribution, features, targets, method='spearman')
print("\nCorrelation for language distributions (Spearman):")
print(correlation_distribution)
# Save the results to a CSV file
correlation_distribution.to_csv(CORELLATION_CROSS_LANGUAGE_PATH, index=False)


##############################################################################################################
# Language-specific Analysis
language_correlations = language_specific_analysis(
df_distribution,
language_column='Language',
feature_columns=features,
target_columns=targets,
method='spearman'
)
language_correlations.to_csv(CORRELATION_LANGUAGE_SPECIFIC_PATH, index=False)
print("All language-specific results have been saved in CSV files.")

