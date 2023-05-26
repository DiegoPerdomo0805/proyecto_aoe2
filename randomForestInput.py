import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt

data = pd.read_csv('aoe_data_clean.csv')

data.drop('map', axis=1, inplace=True)

data.head()

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train[:, :3])
X_test = scaler.transform(X_test[:, :3])

param_grid = {
    'n_estimators': [10, 50, 100],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

classifier = RandomForestClassifier(random_state=0)
grid_search = GridSearchCV(classifier, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Make predictions on the test set
y_pred = grid_search.predict(X_test)
confusion_mat = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Best parameters:", grid_search.best_params_)
print("Confusion Matrix:\n", confusion_mat)
print("Accuracy:", accuracy)

# Get feature importances
importance_scores = grid_search.best_estimator_.feature_importances_
sorted_indices = importance_scores.argsort()[::-1]
top_indices = sorted_indices[:3]
top_five_variables = data.columns[top_indices]
print("Top five indices:\n", sorted_indices)
print("Top five variables:\n", top_five_variables)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(len(importance_scores)), importance_scores[sorted_indices])
plt.xticks(range(len(importance_scores)),
           data.columns[sorted_indices], rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance Scores')
plt.title('Feature Importances')
plt.tight_layout()
plt.show()

menu = True
while menu:

    print("1. Predict")
    print("2. Exit")

    choice = input("Enter your choice: ")

    if choice == '1':

        # User inputs
        # 1104.0,38,25,0
        print('Input the elo')
        user_elo = input()

        print('Input user 1 civ')
        user_p1_civ = input()

        print('Input user 2 civ')
        user_p2_civ = input()

        # Obtain the best estimator from the grid search
        best_estimator = grid_search.best_estimator_

        # Preprocess the user inputs
        # Create a 2D array
        user_input = np.array([[user_elo, user_p1_civ, user_p2_civ]])
        scaled_user_input = scaler.transform(user_input)

        # Make predictions on the scaled user inputs
        predicted_probabilities = best_estimator.predict_proba(
            scaled_user_input)

        # Extract the probability of the winner being 1
        probability_winner_1 = predicted_probabilities[0, 1]

        print("Probability of win:", probability_winner_1)

    else:  # Exit
        menu = False
