import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import re
from math import pow

# Load the dataset
file_path = 'nutrition_cf - Sheet4.csv'
nutrition_data = pd.read_csv(file_path)

nutrition_data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Function to filter dataset based on multiple allergies, region, and category using regex
def filter_dataset(df, allergies, region_pattern, category):
    if allergies == ["no-allergies"]:
        allergy_condition = True
    else:
        allergy_condition = df['Allergy'].apply(lambda x: not any(re.search(allergy, x, re.IGNORECASE) for allergy in allergies))

    if category == 'veg':
        category_condition = df['Category'].str.lower() == 'veg'
    elif category == 'eggetarian':
        category_condition = df['Category'].str.lower().isin(['veg', 'egg'])
    elif category == 'non-veg':
        category_condition = df['Category'].str.lower().isin(['veg', 'egg', 'non-veg'])
    else:
        category_condition = False

    if region_pattern != "Null":
        region_filter = df['Region'].apply(lambda x: re.search(region_pattern, x, re.IGNORECASE) is not None)
    else:
        return df[allergy_condition & category_condition]

    return df[allergy_condition & region_filter & category_condition]

# Function to set target nutritional values based on BMI and gender
def set_target_values(bmi, gender):
    if gender.lower() == 'male':
        if bmi < 18.5:
            return {
                'breakfast': {'Proteins': 15, 'Carbohydrates': 70, 'Fats': 15, 'Fiber': 8, 'Energy(kcal)': 500},
                'lunch': {'Proteins': 18, 'Carbohydrates': 75, 'Fats': 18, 'Fiber': 8, 'Energy(kcal)': 600},
                'snacks': {'Proteins': 10, 'Carbohydrates': 40, 'Fats': 10, 'Fiber': 4, 'Energy(kcal)': 300},
                'dinner': {'Proteins': 20, 'Carbohydrates': 90, 'Fats': 18, 'Fiber': 10, 'Energy(kcal)': 650},
                'appetizers': {'Proteins': 7, 'Carbohydrates': 35, 'Fats': 7, 'Fiber': 5, 'Energy(kcal)': 250},
            }
        elif bmi < 24.9:
            return {
                'breakfast': {'Proteins': 12, 'Carbohydrates': 60, 'Fats': 12, 'Fiber': 6, 'Energy(kcal)': 450},
                'lunch': {'Proteins': 13, 'Carbohydrates': 60, 'Fats': 13, 'Fiber': 6, 'Energy(kcal)': 450},
                'snacks': {'Proteins': 7, 'Carbohydrates': 35, 'Fats': 7, 'Fiber': 3, 'Energy(kcal)': 250},
                'dinner': {'Proteins': 17, 'Carbohydrates': 80, 'Fats': 13, 'Fiber': 8, 'Energy(kcal)': 550},
                'appetizers': {'Proteins': 5, 'Carbohydrates': 30, 'Fats': 5, 'Fiber': 3, 'Energy(kcal)': 200},
            }
        else:
            return {
                'breakfast': {'Proteins': 10, 'Carbohydrates': 50, 'Fats': 10, 'Fiber': 5, 'Energy(kcal)': 400},
                'lunch': {'Proteins': 12, 'Carbohydrates': 50, 'Fats': 12, 'Fiber': 5, 'Energy(kcal)': 400},
                'snacks': {'Proteins': 5, 'Carbohydrates': 30, 'Fats': 5, 'Fiber': 2, 'Energy(kcal)': 200},
                'dinner': {'Proteins': 15, 'Carbohydrates': 70, 'Fats': 10, 'Fiber': 7, 'Energy(kcal)': 500},
                'appetizers': {'Proteins': 4, 'Carbohydrates': 25, 'Fats': 4, 'Fiber': 2, 'Energy(kcal)': 150},
            }
    else:
        if bmi < 18.5:
            return {
                'breakfast': {'Proteins': 14, 'Carbohydrates': 65, 'Fats': 14, 'Fiber': 7, 'Energy(kcal)': 480},
                'lunch': {'Proteins': 16, 'Carbohydrates': 70, 'Fats': 16, 'Fiber': 7, 'Energy(kcal)': 550},
                'snacks': {'Proteins': 9, 'Carbohydrates': 38, 'Fats': 9, 'Fiber': 4, 'Energy(kcal)': 280},
                'dinner': {'Proteins': 18, 'Carbohydrates': 85, 'Fats': 16, 'Fiber': 9, 'Energy(kcal)': 600},
                'appetizers': {'Proteins': 6, 'Carbohydrates': 33, 'Fats': 6, 'Fiber': 4, 'Energy(kcal)': 230},
            }
        elif bmi < 24.9:
            return {
                'breakfast': {'Proteins': 12, 'Carbohydrates': 60, 'Fats': 12, 'Fiber': 6, 'Energy(kcal)': 450},
                'lunch': {'Proteins': 13, 'Carbohydrates': 60, 'Fats': 13, 'Fiber': 6, 'Energy(kcal)': 450},
                'snacks': {'Proteins': 7, 'Carbohydrates': 35, 'Fats': 7, 'Fiber': 3, 'Energy(kcal)': 250},
                'dinner': {'Proteins': 17, 'Carbohydrates': 80, 'Fats': 13, 'Fiber': 8, 'Energy(kcal)': 550},
                'appetizers': {'Proteins': 5, 'Carbohydrates': 30, 'Fats': 5, 'Fiber': 3, 'Energy(kcal)': 200},
            }
        else:
            return {
                'breakfast': {'Proteins': 10, 'Carbohydrates': 50, 'Fats': 10, 'Fiber': 5, 'Energy(kcal)': 400},
                'lunch': {'Proteins': 12, 'Carbohydrates': 50, 'Fats': 12, 'Fiber': 5, 'Energy(kcal)': 400},
                'snacks': {'Proteins': 5, 'Carbohydrates': 30, 'Fats': 5, 'Fiber': 2, 'Energy(kcal)': 200},
                'dinner': {'Proteins': 15, 'Carbohydrates': 70, 'Fats': 10, 'Fiber': 7, 'Energy(kcal)': 500},
                'appetizers': {'Proteins': 4, 'Carbohydrates': 25, 'Fats': 4, 'Fiber': 2, 'Energy(kcal)': 150},
            }

# User inputs for allergies, region, and category
user_allergies_input = input("Enter allergies separated by commas (e.g., mushroom, peanut): ")
user_region_pattern = input("Enter region pattern (regex) (e.g., North): ")
user_category = input("Enter category (e.g., veg, non-veg, eggetarian): ")
user_gender = input("Enter your gender: ")
user_height = float(input("Enter your height: "))
user_weight = int(input("Enter your weight: "))

print("User preference\nAllergy : "+user_allergies_input+"\nRegion : "+user_region_pattern+"\nCategory : "+user_category+"\n")

# Parsing user allergies input into a list
user_allergies = [allergy.strip() for allergy in user_allergies_input.split(',')]
if user_allergies == ['']:
    user_allergies = ["no-allergies"]

# Define associativity rules
associativity_rules = {
    # Your existing associativity rules
}

bmi = (user_weight)/pow(user_height,2)
target_values = set_target_values(bmi, user_gender)

target_breakfast = target_values['breakfast']
target_lunch = target_values['lunch']
target_snacks = target_values['snacks']
target_dinner = target_values['dinner']
target_appetizers = target_values['appetizers']

# Filter dataset
filtered_data = filter_dataset(nutrition_data, user_allergies, user_region_pattern, user_category)

# Define features and target
features = ['Proteins', 'Carbohydrates', 'Fats', 'Fiber', 'Energy(kcal)']
X = filtered_data[features]
y = filtered_data['Food']  # Assuming 'DishName' or another column for classification

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the parameter grid for Grid Search
param_grid = {
    'n_neighbors': [3, 5, 7, 10],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]  # 1 for Manhattan distance, 2 for Euclidean distance
}

# Setup GridSearchCV
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print best parameters
print("Best parameters found: ", grid_search.best_params_)

# Evaluate the best model
best_model = grid_search.best_estimator_
score = best_model.score(X_test, y_test)
print("Test accuracy of best model: ", score)
