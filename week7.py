# Import necessary libraries
import matplotlib.pyplot as plt  # Library for plotting
import numpy as np  # Library for numerical computations
from matplotlib.colors import ListedColormap  # Utility for creating color maps

# Import scikit-learn modules
from sklearn.datasets import make_circles, make_classification, make_moons  # Functions to generate synthetic datasets
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis  # Quadratic Discriminant Analysis classifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier  # Ensemble classifiers
from sklearn.gaussian_process import GaussianProcessClassifier  # Gaussian Process classifier
from sklearn.gaussian_process.kernels import RBF  # Radial Basis Function kernel for Gaussian Process
from sklearn.inspection import DecisionBoundaryDisplay  # Display decision boundaries
from sklearn.model_selection import train_test_split  # Split dataset into train and test sets
from sklearn.naive_bayes import GaussianNB  # Gaussian Naive Bayes classifier
from sklearn.neighbors import KNeighborsClassifier  # K-Nearest Neighbors classifier
from sklearn.neural_network import MLPClassifier  # Multi-layer Perceptron classifier
from sklearn.pipeline import make_pipeline  # Construct pipelines
from sklearn.preprocessing import StandardScaler  # Standardize features by removing mean and scaling to unit variance
from sklearn.svm import SVC  # Support Vector Classification
from sklearn.tree import DecisionTreeClassifier  # Decision Tree classifier

# Define names of classifiers
names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

# Define classifier instances
classifiers = [
    KNeighborsClassifier(3),  # K-Nearest Neighbors with 3 neighbors
    SVC(kernel="linear", C=0.025, random_state=42),  # Linear Support Vector Classifier
    SVC(gamma=2, C=1, random_state=42),  # RBF Support Vector Classifier
    GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),  # Gaussian Process Classifier with RBF kernel
    DecisionTreeClassifier(max_depth=5, random_state=42),  # Decision Tree Classifier with max depth of 5
    RandomForestClassifier(
        max_depth=5, n_estimators=10, max_features=1, random_state=42
    ),  # Random Forest Classifier with max depth of 5, 10 estimators, and maximum features of 1
    MLPClassifier(alpha=1, max_iter=1000, random_state=42),  # Multi-layer Perceptron Classifier with alpha 1 and maximum iterations 1000
    AdaBoostClassifier(algorithm="SAMME", random_state=42),  # AdaBoost Classifier
    GaussianNB(),  # Gaussian Naive Bayes Classifier
    QuadraticDiscriminantAnalysis(),  # Quadratic Discriminant Analysis Classifier
]

# Generate synthetic linearly separable dataset
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

# Define datasets
datasets = [
    make_moons(noise=0.3, random_state=0),  # Generate moon-shaped dataset with noise
    make_circles(noise=0.2, factor=0.5, random_state=1),  # Generate circular dataset with noise
    linearly_separable,  # Use the synthetic linearly separable dataset
]

# Create a figure for plotting
figure = plt.figure(figsize=(27, 9))
i = 1

# Iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # Preprocess dataset: split into training and test sets
    X, y = ds
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    # Determine range for plotting
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # Plot the dataset
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
    ax.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # Iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

        # Create pipeline: StandardScaler and classifier
        clf = make_pipeline(StandardScaler(), clf)
        # Train classifier
        clf.fit(X_train, y_train)
        # Calculate accuracy score
        score = clf.score(X_test, y_test)
        # Display decision boundary
        DecisionBoundaryDisplay.from_estimator(
            clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5
        )

        # Plot training and test points
        ax.scatter(
            X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
        )
        ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c=y_test,
            cmap=cm_bright,
            edgecolors="k",
            alpha=0.6,
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(
            x_max - 0.3,
            y_min + 0.3,
            ("%.2f" % score).lstrip("0"),
            size=15,
            horizontalalignment="right",
        )
        i += 1

# Adjust layout and display plot
plt.tight_layout()
plt.show()
