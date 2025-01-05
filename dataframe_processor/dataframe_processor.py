# data_processor.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataFrameProcessor:
    """
    A class to preprocess a DataFrame, including handling duplicate rows, missing values, 
    outliers, and encoding categorical variables.

    This class provides methods for:
      - Convert neccesary datetime type to pandas datetime objects
      - Removing unnecessary variables
      - Handling duplicate rows and missing values
      - Detecting and handling outliers in quantitative variables using the IQR method
      - Encoding binary and non-binary categorical variables with label and one-hot encoding

    Parameters:
    - df (pd.DataFrame): The input DataFrame to be processed.

    Methods:
    - convert_to_datetime(self, datetime_columns=[]) : Convert specific variables into pd datetime objects.
    - remove_columns(columns_to_remove=[]): Removes specified columns from the DataFrame.
    - handle_duplicate_na(): Processes duplicate rows, handles missing values, 
      and imputes missing data in categorical and numerical variables.
    - handle_quantitative_outliers(): Identifies and caps outliers in quantitative variables.
    - process_encoding(binary_mapping=None, drop_threshold=5): 
      Encodes binary and non-binary categorical variables, and drops high-dimensional variables.
    - get_processed_df(): Returns the processed DataFrame after applying all transformations.

    Example:
        # Load sample data
        df = pd.read_csv('teleco_market_basket.csv').copy()

        # Define binary mapping before creating instance of DataFrameProcessor
        yes_no_mapping = {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 'True': 1, 'False': 0, 'true': 1, 'false': 0}
        
        # Create an instance of DataFrameProcessor
        processor = DataFrameProcessor(df)
        
        # Remove unnecessary columns
        processor.remove_columns(['CaseOrder', 'Customer_id', 'UID', 'Interaction', 'City', 'State', 'County', 'Zip', 'Lat', 'Lng'])
        
        # Run processing methods
        processor.handle_duplicate_na()
        processor.handle_quantitative_outliers()

        processor.process_encoding(binary_mapping=yes_no_mapping)
        
        # Retrieve the processed DataFrame
        df_processed = processor.get_processed_df()
        print("\nProcessed DataFrame:")
        print(df_processed.info())
    """
    def __init__(self, df):
        """
        Initialize the DataFrameProcessor with a DataFrame.
        Input values for columns_exclude will automically import from handle_duplicate_na function 
        
        Parameters:
        - df (DataFrame): The input DataFrame to process.
        """
        self.df = df
        # Replace spaces with underscores and lowercase column names
        df.columns = df.columns.str.replace(" ", "_").str.lower()
        self.columns_exclude = []
        print(f"Processed column names: {list(self.df.columns)}")
        
    def convert_to_datetime(self, datetime_columns=[]):
        """
        Convert specified variables to datetime format.

        Parameters:
        - datetime_columns (list): A list of column names to convert to datetime.
        """
        if datetime_columns:
            existing_columns_to_convert = [col for col in datetime_columns if col in self.df.columns]
            if existing_columns_to_convert:
                for col in existing_columns_to_convert:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                print(f"\nConverted columns to datetime: {existing_columns_to_convert}")
            else:
                print("\nNo valid columns found for datetime conversion.")
        else:
            print("\nNo columns specified for datetime conversion.")
    
    
        
    def remove_columns(self, columns_to_remove=[]):
        """
            Remove unnecessary columns from the DataFrame.

            Parameters:
            - columns_to_remove (list): A list of column names to drop from the DataFrame.
        """
        if columns_to_remove:
            existing_columns_to_remove = [col for col in columns_to_remove if col in self.df.columns]
            if existing_columns_to_remove:
                self.df = self.df.drop(columns=existing_columns_to_remove)
                print(f"\nRemoved unnecessary columns: {existing_columns_to_remove}")
            else:
                print("\nNo unnecessary columns found to remove.")
        else:
            print("\nNo columns specified for removal.")

    def handle_duplicate_na(self, percentage=5):
        """
        Processes duplicate rows, handles missing values, and imputes categorical and numerical variables.
        """
        # Check and remove duplicate rows
        duplicate_rows = self.df.duplicated().sum()
        print(f"Number of duplicate rows: {duplicate_rows}")
        self.df = self.df.drop_duplicates()
        print(f"Number of duplicate rows after removal: {self.df.duplicated().sum()}")

        # Calculate and display initial percentage of missing values
        missing_percentages_initial = self.df.isnull().mean().round(4).mul(100).sort_values(ascending=False)
        print("\nInitial percentage of missing values:")
        print(missing_percentages_initial)

        # Drop rows with missing values in columns with less than 5% missing data
        columns_to_drop_na = missing_percentages_initial[missing_percentages_initial < percentage].index
        self.df = self.df.dropna(subset=columns_to_drop_na)
        self.columns_exclude = list(columns_to_drop_na)
        print(f"\nRows with missing values in {columns_to_drop_na} with less than 5% missing data were dropped.")

        # Updated percentage of missing values
        missing_percentages_updated = self.df.isnull().mean().round(4).mul(100).sort_values(ascending=False)
        print("\nUpdated percentage of missing values:")
        print(missing_percentages_updated,'\n')

        # Impute categorical variables with mode
        for col in self.df.select_dtypes(include=['object', 'category']).columns:
            if self.df[col].isna().sum() > 0:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
                print(f"Imputed missing values in categorical column '{col}' with mode: {self.df[col].mode()[0]}")

        # Impute numerical variables based on skewness
        for col in self.df.select_dtypes(include=['int', 'float']).columns:
            if self.df[col].isna().sum() > 0:
                skewness = self.df[col].skew()
                if abs(skewness) < 0.5:
                    mean_value = self.df[col].mean()
                    self.df[col] = self.df[col].fillna(mean_value)
                    print(f"Imputed '{col}' with mean ({mean_value}) due to low skewness ({skewness:.2f}).")
                else:
                    median_value = self.df[col].median()
                    self.df[col] = self.df[col].fillna(median_value)
                    print(f"Imputed '{col}' with median ({median_value}) due to high skewness ({skewness:.2f}).")

        # Final check for missing values
        print("\nFinal check for missing values in the entire DataFrame:")
        print(self.df.isna().sum().sort_values(ascending=False))

    def handle_quantitative_outliers(self):
        """
        Identifies, visualizes, and caps outliers in quantitative variables using the IQR method.
        """
        quant_variables = self.df.select_dtypes(include=['int', 'float']).columns.tolist()
        if not quant_variables:
            print("\nNo quantitative variables detected; outliers cannot be processed.")
            return

        # Identify variables with outliers using IQR
        variables_with_outliers = []
        for col in quant_variables:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            if not self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)].empty:
                variables_with_outliers.append(col)
        print("\nvariables with outliers:", variables_with_outliers)

        # Visualize the outliers with box plots before handling
        print("\nBox plots for quantitative variables (before handling outliers):")
        self.df[quant_variables].boxplot(rot=45, fontsize=10, figsize=(15, 8))
        plt.title('Box Plots for Quantitative Data (Before Handling Outliers)')
        plt.show()

        # Cap outliers within IQR bounds
        for col in quant_variables:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)

        # Visualize the outliers with box plots after handling
        print("\nBox plots for quantitative variables (after handling outliers):")
        self.df[quant_variables].boxplot(rot=45, fontsize=10, figsize=(15, 8))
        plt.title('Box Plots for Quantitative Data (After Handling Outliers)')
        plt.show()


    def process_encoding(self, binary_mapping=None, drop_threshold=5, cardinality_threshold=100):
        """
        Handles binary and non-binary categorical variables, applies one-hot encoding, and drops high-dimensional variables.
        
        Parameters:
        - binary_mapping (dict): Dictionary for mapping binary values, e.g., {'Yes': 1, 'No': 0}.
        - drop_threshold (int): Minimum number of unique values for non-binary variables to keep for one-hot encoding.
        - cardinality_threshold (int): Maximum number of unique values allowed for a column to be one-hot encoded.
        """
        # Step 1: Select relevant columns
        excluded_id_columns = [col for col in self.df.columns if col.lower().endswith("id")]
        if excluded_id_columns:
            print(f"Function will not encode columns ending with 'id': {excluded_id_columns}")
        selected_variables = [col for col in self.df.columns if col not in self.columns_exclude and col not in excluded_id_columns]
        binary_variables = [col for col in selected_variables if len(self.df[col].unique()) == 2]

        # Step 2: Handle binary variables with label encoding
        if binary_variables and binary_mapping:
            self.df[binary_variables] = self.df[binary_variables].replace(binary_mapping)
            print("\nApplied label encoding to binary variables.")
        else:
            print("\nNo binary variables for label encoding found.")

        # Step 3: Filter non-binary variables
        nonbinary_variables = list(set(selected_variables) - set(binary_variables))
        nonbinary_variables_filter = []
        print('\nNon-binary variable unique values count:')
        for var in nonbinary_variables:
            count = len(self.df[var].unique())
            print(f"{var}: {count}")
            if count <= drop_threshold:
                nonbinary_variables_filter.append(var)
                print(f"after drop threshold: {var}: {count}")

        # Step 4: Drop high-cardinality columns
        high_cardinality_columns = [
        col for col in nonbinary_variables_filter
        if len(self.df[col].unique()) > cardinality_threshold
            ]
        self.df = self.df.drop(columns=high_cardinality_columns)
        print("\nDropped high-cardinality columns:", high_cardinality_columns)

        # Step 5: Drop high-dimensional categorical variables
        cat_drop = [var for var in nonbinary_variables if var not in nonbinary_variables_filter]
        self.df = self.df.drop(columns=cat_drop)
        print("\nDropped high-dimensional categorical variables:", cat_drop)
    
        # Step 6: Log filtered variable counts
        print(f'\nVariable unique values count after unique threshold # {drop_threshold}')
        for var in nonbinary_variables_filter:
            if var not in high_cardinality_columns:  # Only print those retained for encoding
                count = len(self.df[var].unique())
                print(var, count)
    
        # Step 7: One-hot encoding for remaining non-binary categorical variables
        categorical_columns = [
            col for col in nonbinary_variables_filter
            if col not in high_cardinality_columns
        ]
        self.df = pd.get_dummies(self.df, columns=categorical_columns, drop_first=True, dtype=np.int8)
        print("\nDataFrame info after one-hot encoding:")
        print(self.df.info())

    
    def process_all(self, datetime_columns=[], columns_to_remove=[], binary_mapping=None, percentage=5, drop_threshold=5, cardinality_threshold=100):
        """
        Runs all processing steps on the DataFrame in sequence:
        - convert variables into datetime objects
        - remove_columns
        - handle_duplicate_na
        - handle_quantitative_outliers
        - process_encoding

        Parameters:
        -datetime_columns (list): List of variables to convert into pd datetime object.
        - columns_to_remove (list): List of variables to remove.
        - binary_mapping (dict): Dictionary for mapping binary values, e.g., {'Yes': 1, 'No': 0}.
        - percentage (int): Threshold percentage for dropping columns with missing values.
        - drop_threshold (int): Threshold for unique values to decide which columns to drop.

        Example:
        # Load sample data
        df = pd.read_csv('teleco_market_basket.csv')

        # Define binary mapping before creating instance of DataFrameProcessor
        yes_no_mapping = {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 'True': 1, 'False': 0, 'true': 1, 'false': 0}

        # Create an instance of DataFrameProcessor
        processor = DataFrameProcessor(df)

        # Run all processing steps at once
        processor.process_all(
        columns_to_remove=['CaseOrder', 'Customer_id', 'UID', 'Interaction', 'City', 'State', 'County', 'Zip', 'Lat', 'Lng'],
        binary_mapping=yes_no_mapping,
        percentage=5,
        drop_threshold=5
            )

        # Retrieve the processed DataFrame
        df_processed = processor.get_processed_df()
        print("\nProcessed DataFrame:")
        print(df_processed.info())
        """
        self.convert_to_datetime(datetime_columns)
        self.remove_columns(columns_to_remove)
        self.handle_duplicate_na(percentage=percentage)
        self.handle_quantitative_outliers()
        self.process_encoding(binary_mapping=binary_mapping, drop_threshold=drop_threshold, cardinality_threshold=cardinality_threshold)
    
    def get_processed_df(self):
        """
        Returns the processed DataFrame after applying the desired methods.
        
        Returns:
        - DataFrame: The processed DataFrame.
        """
        return self.df