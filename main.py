import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# dataset being used: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

seed = 99

###########################################
### Ingest Data ###
###########################################

# read in the CSV file
data = pd.read_csv("data/breast_cancer_wisconsin.csv")

# separate the class, y, data
y = data["diagnosis"]

# separate the non-class, x, data

x = data.drop(
    ["diagnosis", "radius_mean", "texture_mean", "smoothness_mean", "id"],
    axis=1,
)


# remove empty column
x = x.iloc[:, :-1]


# Introduce missingness
def introduce_missingness(df, missing_percentage=0.3):
    df_with_missing = df.copy()
    n_total_values = df.size  # Total number of values in the DataFrame
    n_missing = int(
        n_total_values * missing_percentage
    )  # Number of values to replace with NaN

    # Randomly select positions in the DataFrame to introduce missing values
    missing_indices = np.random.choice(
        df_with_missing.size, n_missing, replace=False
    )
    row_indices, col_indices = np.unravel_index(
        missing_indices, df_with_missing.shape
    )

    # Assign NaN to the selected indices
    df_with_missing.values[row_indices, col_indices] = np.nan
    return df_with_missing


# Add Gaussian noise
x += np.random.normal(0, 0.3, x.shape)

# Apply the function
x = introduce_missingness(x, missing_percentage=0.5)


# Seed for reproducibility
np.random.seed(seed)

# Encode labels using sklearn's LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # "M" -> 0, "B" -> 1 (or similar)

print(x)


###########################################
### Create PCA for each "fake" CV fold ###
###########################################


def create_pcas(x, y):

    # Define the number of splits
    n_splits = 5

    # Create an array of the row indices
    indices = np.arange(0, len(x))

    # Shuffle order of indices for randomness
    np.random.shuffle(indices)

    # Split the indices
    split_indices = np.array_split(indices, n_splits)

    # Create an imputer and scaler to handle missing values
    imputer = SimpleImputer(
        strategy="mean"
    )  # Replace missing values with column mean
    scaler = StandardScaler()  # Standardize features to mean=0, std=1

    # Define color map for binary classes
    color_map = {0: "orange", 1: "purple"}

    # Perform PCA and plot for each split
    for split_idx, train_indices in enumerate(split_indices):
        train_data = x.iloc[train_indices]
        train_labels = y[train_indices]

        # Impute missing values
        train_data = imputer.fit_transform(train_data)

        # Standardize the data
        train_data = scaler.fit_transform(train_data)

        # Fit PCA on the training data
        pca = PCA(n_components=2)  # Only 2 components in a plot
        pca.fit(train_data)
        transformed_data = pca.fit_transform(train_data)
        explained_variance = (
            pca.explained_variance_ratio_ * 100
        )  # Percentage explained

        # Create a scatter plot
        plt.figure(figsize=(8, 6))
        colours = [
            color_map[label] for label in train_labels
        ]  # Map labels to colors
        scatter = plt.scatter(
            transformed_data[:, 0],
            transformed_data[:, 1],
            c=colours,  # Use class labels for coloring
            alpha=0.7,
            edgecolor="k",
        )
        # Add a legend
        for label, color in color_map.items():
            # Use inverse_transform to display original labels
            plt.scatter(
                [],
                [],
                color=color,
                label=f"Class {label_encoder.inverse_transform([label])[0]}",
            )

        plt.legend(title="Class Labels")
        plt.title(f"PCA Plot - Split {split_idx + 1}")
        plt.xlabel(
            f"Principal Component 1 ({explained_variance[0]:.2f}% Variance Explained)"
        )
        plt.ylabel(
            f"Principal Component 2 ({explained_variance[1]:.2f}% Variance Explained)"
        )
        plt.grid(True)

        # Save the plot to a file
        filename = f"outputs/pca_plot_split_{split_idx + 1}.png"
        plt.savefig(filename, dpi=300)
        plt.close()  # Close the figure to free memory

        print(f"PCA plot for Split {split_idx + 1} saved as {filename}")


###########################################
### Run SVM with data leakage ###
###########################################


def data_leak_anal(x, y):
    # Pre-processing
    imputer = SimpleImputer(strategy="mean")  # Handle missing values
    scaler = StandardScaler()  # Standardize the data

    x_imputed = imputer.fit_transform(x)  # Impute missing values
    x_scaled = scaler.fit_transform(x_imputed)  # Scale the data

    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # Metrics for each fold
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    # Run cross-validation
    for train_idx, test_idx in cv.split(x_scaled, y):
        x_train, x_test = x_scaled[train_idx], x_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train SVM
        svm = SVC(C=1, kernel="linear", random_state=seed)
        svm.fit(x_train, y_train)

        # Predict on test set
        y_pred = svm.predict(x_test)

        # Calculate metrics
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        precision_scores.append(
            precision_score(y_test, y_pred, average="weighted")
        )
        recall_scores.append(recall_score(y_test, y_pred, average="weighted"))
        f1_scores.append(f1_score(y_test, y_pred, average="weighted"))

    # Print results
    print("SVM with data leakage:")
    print(f"Accuracy scores: {accuracy_scores}")
    print(f"Precision scores: {precision_scores}")
    print(f"Recall scores: {recall_scores}")
    print(f"F1 scores: {f1_scores}")
    print(f"Mean Accuracy: {np.mean(accuracy_scores):.2f}")
    print(f"Mean Precision: {np.mean(precision_scores):.2f}")
    print(f"Mean Recall: {np.mean(recall_scores):.2f}")
    print(f"Mean F1 Score: {np.mean(f1_scores):.2f}")


data_leak_anal(x, y)

###########################################
### Run SVM with no data leakage ###
###########################################


def run_correct_anal(x, y):
    # Cross-validation setup
    imputer = SimpleImputer(strategy="mean")  # Handle missing values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # Metrics for each fold
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    # Run cross-validation
    for train_idx, test_idx in cv.split(x, y):
        # Use .iloc for row-based indexing
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Define the pipeline
        pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="mean"),
                ),  # Handle missing values
                ("scaler", StandardScaler()),  # Standardize the data
                (
                    "svm",
                    SVC(C=1, kernel="linear", random_state=seed),
                ),  # SVM Classifier
            ]
        )

        # Fit the pipeline
        pipeline.fit(x_train, y_train)

        # Predict on test set
        y_pred = pipeline.predict(x_test)

        # Calculate metrics
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        precision_scores.append(
            precision_score(y_test, y_pred, average="weighted")
        )
        recall_scores.append(recall_score(y_test, y_pred, average="weighted"))
        f1_scores.append(f1_score(y_test, y_pred, average="weighted"))

    # Print results
    print("\nSVM with no data leakage:")
    print(f"Accuracy scores: {accuracy_scores}")
    print(f"Precision scores: {precision_scores}")
    print(f"Recall scores: {recall_scores}")
    print(f"F1 scores: {f1_scores}")
    print(f"Mean Accuracy: {np.mean(accuracy_scores):.2f}")
    print(f"Mean Precision: {np.mean(precision_scores):.2f}")
    print(f"Mean Recall: {np.mean(recall_scores):.2f}")
    print(f"Mean F1 Score: {np.mean(f1_scores):.2f}")


run_correct_anal(x, y)
