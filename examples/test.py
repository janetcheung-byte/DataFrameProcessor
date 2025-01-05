import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dataframe_processor import DataFrameProcessor
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)


# Load sample data
df = pd.read_csv('test.csv').copy()

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