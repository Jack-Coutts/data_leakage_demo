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
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
)
from sklearn.metrics import f1_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.impute import KNNImputer

# dataset being used: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

###########################################
### Ingest Data ###
###########################################

# read in the CSV file
data = pd.read_csv("data/breast_cancer_wisconsin.csv")

# separate the class, y, data
y = data["diagnosis"]

# separate the non-class, x, data
"""
x = data.drop(
    ["diagnosis", "radius_mean", "texture_mean", "smoothness_mean", "id"],
    axis=1,
)
"""

x = data[["perimeter_mean", "area_mean", "compactness_mean"]]

# remove empty column
# x = x.iloc[:, :-1]


# Introduce 30% missingness
def introduce_missingness(df, missing_percentage=0.3):
    """
    Introduce missing values randomly into a DataFrame.

    Parameters:
    - df: pd.DataFrame, the input DataFrame.
    - missing_percentage: float, the fraction of missing values to introduce.

    Returns:
    - df_with_missing: pd.DataFrame with missing values.
    """
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
# x += np.random.normal(0, 0.1, x.shape)

# Apply the function
x = introduce_missingness(x, missing_percentage=0.7)


# Seed for reproducibility
np.random.seed(42)

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

    # Custom Scorer (F1 Score for Classification)
    f1_scorer = make_scorer(f1_score, average="weighted")

    # Grid Search for SVM Hyperparameters
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_score = -1
    best_params = None

    # Define hyperparameter grid
    param_grid = {
        "C": [0.1, 1, 10, 100],  # Regularization parameter
        "kernel": ["linear"],  # Kernel types
        "gamma": ["scale", "auto"],  # Kernel coefficient for rbf
    }

    # Grid Search Loop
    for C in param_grid["C"]:
        for kernel in param_grid["kernel"]:
            for gamma in param_grid["gamma"]:
                svm = SVC(C=C, kernel=kernel, gamma=gamma, random_state=42)
                scores = cross_val_score(
                    svm, x_scaled, y, cv=cv, scoring=f1_scorer
                )
                mean_score = np.mean(scores)

                if mean_score > best_score:
                    best_score = mean_score
                    best_params = {
                        "C": C,
                        "kernel": kernel,
                        "gamma": gamma,
                    }

    # Output the best parameters and F1 score
    print("SVM with data leakage: ")
    print(f"Best Parameters: {best_params}")
    print(f"Best Mean F1 Score: {best_score:.2f}")


data_leak_anal(x, y)

###########################################
### Run SVM with no data leakage ###
###########################################


def run_correct_anal(x, y):

    imputer = SimpleImputer(strategy="mean")  # Handle missing values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Custom Scorer (F1 Score for Classification)
    f1_scorer = make_scorer(f1_score, average="weighted")

    # Define the Pipeline
    pipeline = Pipeline(
        [
            ("imputer", imputer),  # Handle missing values
            ("scaler", StandardScaler()),  # Standardize the data
            ("svm", SVC(random_state=42)),  # SVM Classifier
        ]
    )

    # Define Hyperparameter Grid
    param_grid = {
        "svm__C": [0.1, 1, 10, 100],
        "svm__kernel": ["linear"],
        "svm__gamma": ["scale", "auto"],
    }

    # Perform Grid Search with Cross-Validation
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=f1_scorer,
        cv=cv,
        n_jobs=-1,  # Parallelize across CPUs
        verbose=1,
    )

    # Fit Grid Search
    grid_search.fit(x, y)

    # Output the best parameters and F1 score
    print("SVM with no data leakage: ")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Mean F1 Score: {grid_search.best_score_:.2f}")
    feature_importances = grid_search.best_estimator_.named_steps["svm"].coef_
    print("Feature Importances:", feature_importances)


run_correct_anal(x, y)
