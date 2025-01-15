import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# dataset being used: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data


# function to print output to the terminal
def print_to_console(input):
    print("----------")
    print(input)
    print(f"The input data type is: {type(input)}")


# read in the Iris CSV file
complete_iris_dataset = pd.read_csv("data/breast_cancer_wisconsin.csv")
print_to_console(complete_iris_dataset)

# separate the class, y, data
iris_class = complete_iris_dataset["diagnosis"]
print_to_console(iris_class)

# separate the non-class, x, data
input_data = complete_iris_dataset.drop("diagnosis", axis=1)
print_to_console(input_data)

# impute
imputer = SimpleImputer(strategy="mean")
input_data = imputer.fit_transform(input_data)

# Step 2: Standardize the data
scaler = StandardScaler()
input_data = scaler.fit_transform(input_data)

# run pca
number_of_components = 3
pca = PCA(n_components=number_of_components)
principal_components = pca.fit_transform(input_data)


# Create a DataFrame with the principal components
principal_df = pd.DataFrame(
    principal_components,
    columns=[f"Principal Component {i+1}" for i in range(number_of_components)],
)
principal_df["Class"] = iris_class

# Step 4: Plot the PCA results
plt.figure(figsize=(8, 6))
classes = np.unique(iris_class)
colors = ["red", "blue"]  # Define colors for classes

for class_label, color in zip(classes, colors):
    indices_to_plot = principal_df["Class"] == class_label
    plt.scatter(
        principal_df.loc[indices_to_plot, "Principal Component 1"],
        principal_df.loc[indices_to_plot, "Principal Component 2"],
        c=color,
        label=f"Class {class_label}",
        s=50,
    )

plt.title("PCA - Colored by Class Labels")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid()
# plt.show()


### PLS-DA


# Step 3: Encode class labels
le = LabelEncoder()
y_encoded = le.fit_transform(iris_class)

comps = 2
pls = PLSRegression(n_components=comps)
pls.fit(input_data, y_encoded)


# Step 5: Get predictions and evaluate
y_pred = np.round(pls.predict(input_data)).astype(int)
accuracy = accuracy_score(y_encoded, y_pred)
precision = precision_score(y_encoded, y_pred, average="weighted")
recall = recall_score(y_encoded, y_pred, average="weighted")
f1 = f1_score(y_encoded, y_pred, average="weighted")

# Print metrics
print("Classification Report:")
# print(
#    classification_report(
# y_encoded, y_pred, target_names=le.classes_.astype(str)
# )
# )
print("Confusion Matrix:")
print(confusion_matrix(y_encoded, y_pred))
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Step 6: Visualize the PLS-DA components
components = pls.x_scores_  # PLS components
plt.figure(figsize=(8, 6))
classes = np.unique(y_encoded)
colors = ["red", "blue"]

for class_label, color in zip(classes, colors):
    indices_to_plot = y_encoded == class_label
    plt.scatter(
        components[indices_to_plot, 0],
        components[indices_to_plot, 1],
        c=color,
        label=f"Class {int(class_label)}",
        s=50,
    )

plt.title("PLS-DA - Colored by Class Labels")
plt.xlabel("PLS Component 1")
plt.ylabel("PLS Component 2")
plt.legend()
plt.grid()
plt.show()
