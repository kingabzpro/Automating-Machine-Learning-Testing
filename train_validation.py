import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import model_evaluation

# Load the loan data from a CSV file
loan_data = pd.read_csv("data/loan_data.csv")

# Define the label column
label_col = "not.fully.paid"

# Train test split
df_train, df_test = train_test_split(
    loan_data, stratify=loan_data[label_col], random_state=0
)

# Encode the 'purpose' column
label_encoder = LabelEncoder()
df_train["purpose"] = label_encoder.fit_transform(df_train["purpose"])
df_test["purpose"] = label_encoder.transform(df_test["purpose"])

# Standardize the features
scaler = StandardScaler()
df_train[df_train.columns.difference([label_col])] = scaler.fit_transform(df_train[df_train.columns.difference([label_col])])
df_test[df_test.columns.difference([label_col])] = scaler.transform(df_test[df_test.columns.difference([label_col])])

# Define models
model_1 = LogisticRegression(random_state=1, max_iter=10000)
model_2 = RandomForestClassifier(n_estimators=50, random_state=1)
model_3 = GaussianNB()

# Create the VotingClassifier
clf_model = VotingClassifier(
    estimators=[("lr", model_1), ("rf", model_2), ("gnb", model_3)], voting="soft"
)

# Train the model
clf_model.fit(df_train.drop(label_col, axis=1), df_train[label_col])

# Calculate the accuracy score using the .score function
accuracy = clf_model.score(df_test.drop(label_col, axis=1), df_test[label_col])

# Print the accuracy score
print(f"Accuracy: {accuracy:.2f}")

# Create Deepchecks datasets
deep_train = Dataset(df_train, label=label_col, cat_features=["purpose"])
deep_test = Dataset(df_test, label=label_col, cat_features=["purpose"])

# Run the evaluation suite
evaluation_suite = model_evaluation()
suite_result = evaluation_suite.run(deep_train, deep_test, clf_model)

# Save the results as HTML
suite_result.save_as_html("results/model_validation.html")
