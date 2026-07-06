
import numpy as np
import pandas as pd
from Machine_Learning_algorithms.GaussianNaiveBayes.model import GaussianNaiveBayesModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib as plt


def main():

    # ============================================
    # Load dataset
    # ============================================

    data = pd.read_csv("DataSets/Fish.csv")

    X = data.drop("Species", axis=1).values
    y = data["Species"].values

    # ============================================
    # Train / Test split
    # ============================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # ============================================
    # Feature Scaling
    # ============================================

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

   

    # Create and train the model
    model = GaussianNaiveBayesModel()
    model.fit(X_train, y_train,loss_fn= None , optimizer= None)

    # Make predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    # Evaluate accuracy
    accuracy = model.score(X_test, y_test)
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()

