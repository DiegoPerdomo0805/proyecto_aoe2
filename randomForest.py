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


def print_civilizations():
    factions = {
        "Vikings": 0,
        "Britons": 1,
        "Chinese": 2,
        "Mayans": 3,
        "Berbers": 4,
        "Khmer": 5,
        "Cumans": 6,
        "Huns": 7,
        "Malay": 8,
        "Ethiopians": 9,
        "Magyars": 10,
        "Franks": 11,
        "Tatars": 12,
        "Slavs": 13,
        "Celts": 14,
        "Mongols": 15,
        "Teutons": 16,
        "Koreans": 17,
        "Aztecs": 18,
        "Goths": 19,
        "Turks": 20,
        "Japanese": 21,
        "Persians": 22,
        "Indians": 23,
        "Saracens": 24,
        "Burgundians": 25,
        "Bulgarians": 26,
        "Byzantines": 27,
        "Lithuanians": 28,
        "Sicilians": 29,
        "Malians": 30,
        "Portuguese": 31,
        "Spanish": 32,
        "Vietnamese": 33,
        "Italians": 34,
        "Burmese": 35,
        "Poles": 36,
        "Incas": 37,
        "Bohemians": 38
    }

    num_civilizations = len(factions)
    col_width = 25
    num_columns = 3
    num_per_column = (num_civilizations + num_columns - 1) // num_columns

    sorted_factions = sorted(factions.keys())
    for i in range(0, num_civilizations, num_per_column):
        for j in range(i, min(i + num_per_column, num_civilizations)):
            faction_name = sorted_factions[j]
            faction_number = factions[faction_name]
            print(f"{faction_name:{col_width}}", end="")
        print()


def get_faction_number(faction_name):
    factions = {
        "Vikings": 0,
        "Britons": 1,
        "Chinese": 2,
        "Mayans": 3,
        "Berbers": 4,
        "Khmer": 5,
        "Cumans": 6,
        "Huns": 7,
        "Malay": 8,
        "Ethiopians": 9,
        "Magyars": 10,
        "Franks": 11,
        "Tatars": 12,
        "Slavs": 13,
        "Celts": 14,
        "Mongols": 15,
        "Teutons": 16,
        "Koreans": 17,
        "Aztecs": 18,
        "Goths": 19,
        "Turks": 20,
        "Japanese": 21,
        "Persians": 22,
        "Indians": 23,
        "Saracens": 24,
        "Burgundians": 25,
        "Bulgarians": 26,
        "Byzantines": 27,
        "Lithuanians": 28,
        "Sicilians": 29,
        "Malians": 30,
        "Portuguese": 31,
        "Spanish": 32,
        "Vietnamese": 33,
        "Italians": 34,
        "Burmese": 35,
        "Poles": 36,
        "Incas": 37,
        "Bohemians": 38
    }

    return factions.get(faction_name, -1)


def get_faction_name(faction_number):
    factions = {
        0: "Vikings",
        1: "Britons",
        2: "Chinese",
        3: "Mayans",
        4: "Berbers",
        5: "Khmer",
        6: "Cumans",
        7: "Huns",
        8: "Malay",
        9: "Ethiopians",
        10: "Magyars",
        11: "Franks",
        12: "Tatars",
        13: "Slavs",
        14: "Celts",
        15: "Mongols",
        16: "Teutons",
        17: "Koreans",
        18: "Aztecs",
        19: "Goths",
        20: "Turks",
        21: "Japanese",
        22: "Persians",
        23: "Indians",
        24: "Saracens",
        25: "Burgundians",
        26: "Bulgarians",
        27: "Byzantines",
        28: "Lithuanians",
        29: "Sicilians",
        30: "Malians",
        31: "Portuguese",
        32: "Spanish",
        33: "Vietnamese",
        34: "Italians",
        35: "Burmese",
        36: "Poles",
        37: "Incas",
        38: "Bohemians"
    }

    return factions.get(faction_number, "Unknown")


def analiseBestOption(user_elo, user_p2_civ):

    resultCiv = 0
    resultPoints = 0
    for user_p1_civ in range(0, 38):
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

        if probability_winner_1 >= resultPoints:
            resultPoints = probability_winner_1
            resultCiv = user_p1_civ

    print('Your best option to win against ', get_faction_name(user_p2_civ), ' is ',
          get_faction_name(resultCiv), ' with ', resultPoints * 100, '% probability of winning')


menu = True
while menu:

    print("1. Predict")
    print("2. Exit")

    choice = input("Enter your choice: ")

    if choice == '1':

        # User inputs
        # 1104.0,38,25,0
        # 1080.0,25,22,1
        print('Input the elo')
        user_elo = input()

        # Call the function to print the civilizations
        print_civilizations()

        print('Input user 1 civ')
        user_p1_civInput = input()

        print('Input user 2 civ')
        user_p2_civInput = input()

        user_p1_civ = get_faction_number(user_p1_civInput)
        user_p2_civ = get_faction_number(user_p2_civInput)

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

        print("Probability of win:", probability_winner_1*100, "%")

        analiseBestOption(user_elo, user_p2_civ)

    elif choice == 2:  # Exit
        menu = False
    else:
        print("Invalid input. Please try again.")
