import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from parallelized_rf import RandomForestClassifier621

# Load the wine dataset
X, y = load_wine(return_X_y=True)

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#run training
if __name__ == "__main__":
    rf1 = RandomForestClassifier621(n_estimators=100, min_samples_leaf=1, max_features=0.4, max_depth=None, label_depth=None, force_label_split=False)
    # Fit the model to the training data
    rf1.fit(X_train, y_train)

    # Make pred ictions on new data
    predictions = rf1.predict(X_test)

    # Calculate the accuracy score for the predictions
    accuracy_score = rf1.score(X_test, y_test)
    print(accuracy_score)