import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity

# Load the loan data from a CSV file
loan_data = pd.read_csv("data/loan_data.csv")

# Define the label column
label_col = "not.fully.paid"

# Create a Deepchecks Dataset object with the loan data
deep_loan_data = Dataset(loan_data, label=label_col, cat_features=["purpose"])

# Initialize the data integrity suite
integ_suite = data_integrity()

# Run the data integrity suite on the dataset
suite_result = integ_suite.run(deep_loan_data)

# Save the results of the data integrity suite as an HTML file
suite_result.save_as_html("results/data_validation.html")
