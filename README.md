# DataFrameProcessor

The `DataFrameProcessor` is a Python class designed to streamline and automate the preprocessing of tabular data in Pandas DataFrames. It simplifies common data-cleaning tasks such as handling missing values, detecting and removing outliers, and encoding categorical variables. This tool is ideal for data analysts and data scientists who need to prepare datasets for machine learning models or other types of analysis.

---

## Features
The `DataFrameProcessor` provides the following functionalities:

1. **Column Standardization**:
   - Automatically converts column names to lowercase and replaces spaces with underscores for consistency.

2. **Datetime Conversion**:
   - Converts specified columns into Pandas datetime objects for easier time-based analysis.

3. **Column Removal**:
   - Allows removal of unnecessary columns based on user-defined lists.

4. **Handling Missing Values**:
   - Removes rows with missing values in columns where the percentage of missing data is below a threshold.
   - Fills missing categorical values with the mode and numeric values with the mean or median, depending on skewness.

5. **Duplicate Removal**:
   - Detects and removes duplicate rows from the dataset.

6. **Outlier Detection and Handling**:
   - Identifies outliers in numerical data using the Interquartile Range (IQR) method.
   - Caps outliers to specified limits to reduce their impact on analysis.

7. **Categorical Variable Encoding**:
   - Automatically applies label encoding to binary categorical variables.
   - One-hot encodes non-binary categorical variables, with options to drop high-cardinality columns.

8. **Customizable Processing Steps**:
   - Offers the flexibility to customize thresholds and select specific preprocessing steps.

9. **Pipeline Execution**:
   - Provides a `process_all()` method to execute all preprocessing steps sequentially.

---



## Key Advantages
- **Automation**: Reduces the need for repetitive data-cleaning scripts.
- **Customizable**: Flexible thresholds and encoding options to suit different datasets.
- **Integration**: Compatible with Pandas, making it easy to integrate into existing workflows.

---

## Limitations
- **Memory Usage**: One-hot encoding may cause dimensionality explosion for high-cardinality categorical variables.
- **Assumptions in Imputation**: Mean/median imputation assumes a certain data distribution, which might not always align with the actual data.

---
## Installation

For Conda environment:
```bash
run in the cmd: conda env create -f environment.yml
```

For pip virtual environment:
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Install Directly from GitHub
```bash
pip install git+https://github.com/janetcheung-byte/DataFrameProcessor.git
```
For extra code example, please check under examples folder.

Code usage example:
```python
import pandas as pd
from dataframe_processor import DataFrameProcessor

# Load your dataset
df = pd.read_csv('example_dataset.csv')

# Create an instance of the processor
processor = DataFrameProcessor(df)

# Define unnecessary columns to remove
columns_to_remove = ['id', 'timestamp', 'zip_code']

# Define binary mapping for encoding
binary_mapping = {'Yes': 1, 'No': 0}

# Run all preprocessing steps
processor.process_all(
    datetime_columns=['order_date', 'delivery_date'],
    columns_to_remove=columns_to_remove,
    binary_mapping=binary_mapping,
    percentage=5,
    drop_threshold=5,
    cardinality_threshold=100
)

# Retrieve the processed DataFrame
processed_df = processor.get_processed_df()

# View the cleaned data
print(processed_df.info())
```






## Contribution
Contributions are welcome! If you’d like to add features, fix bugs, or improve documentation, feel free to fork this repository and submit a pull request.

---

## Support the Project ❤️

If you find this library helpful in your projects, please consider giving it a ⭐ on GitHub!  
It motivates me to continue improving and adding more features.

[![GitHub Stars](https://img.shields.io/github/stars/janetcheung-byte/DataFrameProcessor?style=social)](https://github.com/janetcheung-byte/DataFrameProcessor)


## License
This project is licensed under the **Apache License 2.0**.

You may obtain a copy of the license at:

http://www.apache.org/licenses/LICENSE-2.0


Unless required by applicable law or agreed to in writing, software distributed under this license is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the license for the specific language governing permissions and limitations under the license.





