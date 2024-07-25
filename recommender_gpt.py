# recommender.py 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import re
from math import pow

# Load the dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to filter dataset based on multiple allergies, region, and category using regex
def filter_dataset(df, allergies, region_pattern, category):
    if allergies == ["no-allergies"]:
        allergy_condition = df['Allergy'].apply(lambda x: True)
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
            # Underweight
            return {
                'breakfast': {'Proteins': 15, 'Carbohydrates': 70, 'Fats': 15, 'Fiber': 8, 'Energy(kcal)': 500},
                'lunch': {'Proteins': 18, 'Carbohydrates': 75, 'Fats': 18, 'Fiber': 8, 'Energy(kcal)': 600},
                'snacks': {'Proteins': 10, 'Carbohydrates': 40, 'Fats': 10, 'Fiber': 4, 'Energy(kcal)': 300},
                'dinner': {'Proteins': 20, 'Carbohydrates': 90, 'Fats': 18, 'Fiber': 10, 'Energy(kcal)': 650},
                'appetizers': {'Proteins': 7, 'Carbohydrates': 35, 'Fats': 7, 'Fiber': 5, 'Energy(kcal)': 250},
            }
        elif bmi < 24.9:
            # Normal weight
            return {
                'breakfast': {'Proteins': 12, 'Carbohydrates': 60, 'Fats': 12, 'Fiber': 6, 'Energy(kcal)': 450},
                'lunch': {'Proteins': 13, 'Carbohydrates': 60, 'Fats': 13, 'Fiber': 6, 'Energy(kcal)': 450},
                'snacks': {'Proteins': 7, 'Carbohydrates': 35, 'Fats': 7, 'Fiber': 3, 'Energy(kcal)': 250},
                'dinner': {'Proteins': 17, 'Carbohydrates': 80, 'Fats': 13, 'Fiber': 8, 'Energy(kcal)': 550},
                'appetizers': {'Proteins': 5, 'Carbohydrates': 30, 'Fats': 5, 'Fiber': 3, 'Energy(kcal)': 200},
            }
        else:
            # Overweight
            return {
                'breakfast': {'Proteins': 10, 'Carbohydrates': 50, 'Fats': 10, 'Fiber': 5, 'Energy(kcal)': 400},
                'lunch': {'Proteins': 12, 'Carbohydrates': 50, 'Fats': 12, 'Fiber': 5, 'Energy(kcal)': 400},
                'snacks': {'Proteins': 5, 'Carbohydrates': 30, 'Fats': 5, 'Fiber': 2, 'Energy(kcal)': 200},
                'dinner': {'Proteins': 15, 'Carbohydrates': 70, 'Fats': 10, 'Fiber': 7, 'Energy(kcal)': 500},
                'appetizers': {'Proteins': 4, 'Carbohydrates': 25, 'Fats': 4, 'Fiber': 2, 'Energy(kcal)': 150},
            }
    else:
        # Similar logic for female, adjust values as needed
        if bmi < 18.5:
            # Underweight
            return {
                'breakfast': {'Proteins': 14, 'Carbohydrates': 65, 'Fats': 14, 'Fiber': 7, 'Energy(kcal)': 480},
                'lunch': {'Proteins': 16, 'Carbohydrates': 70, 'Fats': 16, 'Fiber': 7, 'Energy(kcal)': 550},
                'snacks': {'Proteins': 9, 'Carbohydrates': 38, 'Fats': 9, 'Fiber': 4, 'Energy(kcal)': 280},
                'dinner': {'Proteins': 18, 'Carbohydrates': 85, 'Fats': 16, 'Fiber': 9, 'Energy(kcal)': 600},
                'appetizers': {'Proteins': 6, 'Carbohydrates': 33, 'Fats': 6, 'Fiber': 4, 'Energy(kcal)': 230},
            }
        elif bmi < 24.9:
            # Normal weight
            return {
                'breakfast': {'Proteins': 12, 'Carbohydrates': 60, 'Fats': 12, 'Fiber': 6, 'Energy(kcal)': 450},
                'lunch': {'Proteins': 13, 'Carbohydrates': 60, 'Fats': 13, 'Fiber': 6, 'Energy(kcal)': 450},
                'snacks': {'Proteins': 7, 'Carbohydrates': 35, 'Fats': 7, 'Fiber': 3, 'Energy(kcal)': 250},
                'dinner': {'Proteins': 17, 'Carbohydrates': 80, 'Fats': 13, 'Fiber': 8, 'Energy(kcal)': 550},
                'appetizers': {'Proteins': 5, 'Carbohydrates': 30, 'Fats': 5, 'Fiber': 3, 'Energy(kcal)': 200},
            }
        else:
            # Overweight
            return {
                'breakfast': {'Proteins': 10, 'Carbohydrates': 50, 'Fats': 10, 'Fiber': 5, 'Energy(kcal)': 400},
                'lunch': {'Proteins': 12, 'Carbohydrates': 50, 'Fats': 12, 'Fiber': 5, 'Energy(kcal)': 400},
                'snacks': {'Proteins': 5, 'Carbohydrates': 30, 'Fats': 5, 'Fiber': 2, 'Energy(kcal)': 200},
                'dinner': {'Proteins': 15, 'Carbohydrates': 70, 'Fats': 10, 'Fiber': 7, 'Energy(kcal)': 500},
                'appetizers': {'Proteins': 4, 'Carbohydrates': 25, 'Fats': 4, 'Fiber': 2, 'Energy(kcal)': 150},
            }

def check_combined_nutritional_requirements(food1, food2, target_nutrients):
    combined_nutrients = food1[['Proteins', 'Carbohydrates', 'Fats', 'Fiber', 'Energy(kcal)']] + food2[['Proteins', 'Carbohydrates', 'Fats', 'Fiber', 'Energy(kcal)']]
    meets_requirements = all(combined_nutrients <= pd.Series(target_nutrients))
    if(combined_nutrients <= pd.Series(target_nutrients)).all():
        return meets_requirements

def check_nutritional_requirements(food, target_nutrients):
    combined_nutrients = food[['Proteins', 'Carbohydrates', 'Fats', 'Fiber', 'Energy(kcal)']]
    meets_requirements = all(combined_nutrients <= pd.Series(target_nutrients))
    if(combined_nutrients <= pd.Series(target_nutrients)).all():
        return meets_requirements

def recommend_food(df, meal_type, target_nutrients, num_recommendations=27):
    meal_data = df[df['Type'].str.contains(meal_type, case=False, na=False)]
    features = ['Proteins', 'Carbohydrates', 'Fats', 'Fiber', 'Energy(kcal)', 'Carbon Footprint(kg CO2e)']

    X = meal_data[features].values
    y = meal_data['Food'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn = NearestNeighbors(n_neighbors=num_recommendations,metric= 'cosine')
    knn.fit(X_train_scaled, y_train)

    target_values = np.array([target_nutrients['Proteins'], target_nutrients['Carbohydrates'],
                              target_nutrients['Fats'], target_nutrients['Fiber'],
                              target_nutrients['Energy(kcal)'], 0]).reshape(1, -1)
    target_values_scaled = scaler.transform(target_values)

    distances, indices = knn.kneighbors(target_values_scaled)

    recommended_foods = meal_data.iloc[indices[0]]
    #print(recommended_foods.head(2))
    return recommended_foods.head(num_recommendations)

'''def prepare_recommendations(df, bmi, gender, allergies, region, category):
    target_values = set_target_values(bmi, gender)
    recommendations = {}
    for meal_type, target_nutrients in target_values.items():
        meal_recommendations, distances = recommend_food(df, meal_type, target_nutrients)
        if len(meal_recommendations) == 0:
            continue
        filtered_recommendations = filter_dataset(meal_recommendations, allergies, region, category)
        filtered_recommendations['Distance'] = distances[:len(filtered_recommendations)]
        recommendations[meal_type] = filtered_recommendations
    return recommendations'''

def divide_by_serving(food):
  serving = food['Serving']
  food['Serving_Numbers'] =  int(serving[0])
  food[['Proteins', 'Carbohydrates', 'Fats', 'Fiber', 'Energy(kcal)']] = food[['Proteins', 'Carbohydrates', 'Fats', 'Fiber', 'Energy(kcal)']].div(food['Serving_Numbers'], axis=0)
  food = food.drop(['Serving_Numbers'], axis=0)
  return food

def divide_by_serving_combo(food, assoc_food, target_nutrients):
    serving = food['Serving']
    assoc_serving = assoc_food['Serving']

    new_food = food.copy()
    new_assoc_food = assoc_food.copy()

    new_food['Serving_Numbers'] = int(serving[0])
    new_assoc_food['Serving_Numbers_assoc'] = int(assoc_serving[0])

    new_food[['Proteins', 'Carbohydrates', 'Fats', 'Fiber', 'Energy(kcal)']] = new_food[['Proteins', 'Carbohydrates', 'Fats', 'Fiber', 'Energy(kcal)']].div(new_food['Serving_Numbers'], axis=0)
    new_assoc_food[['Proteins', 'Carbohydrates', 'Fats', 'Fiber', 'Energy(kcal)']] = new_assoc_food[['Proteins', 'Carbohydrates', 'Fats', 'Fiber', 'Energy(kcal)']].div(new_assoc_food['Serving_Numbers_assoc'], axis=0)

    if check_combined_nutritional_requirements(new_food, new_assoc_food, target_nutrients):
        return [new_food['Food'], new_food['Energy(kcal)'], new_assoc_food["Food"], new_assoc_food["Energy(kcal)"]]

    new_food = new_food.drop(['Serving_Numbers'], axis=0)
    new_assoc_food = new_assoc_food.drop(['Serving_Numbers_assoc'], axis=0)

    return [new_food['Food'], new_food['Energy(kcal)']]

def get_weekly_plan(recommended_foods, associative_rules, valid_associations, target_nutrients):
    recommendations_with_associations = []

    for index, row in recommended_foods.iterrows():
        food_item = row['Food']
        associativity = str(row['Associativity'])
        carbon_footprint = row['Carbon Footprint(kg CO2e)']
        calorie = row['Energy(kcal)']

        # Split associativity values if they are combined (e.g., '2,8')
        associativity_values = [value.strip() for value in associativity.split(',')]
        associated_foods = []
        cal = []

        for value in associativity_values:
            if value in valid_associations and value in associative_rules:
                associated_food_items = recommended_foods[recommended_foods['Associativity'].isin(associative_rules[value])]
                if not associated_food_items.empty:
                    for _, assoc_row in associated_food_items.iterrows():
                        if check_combined_nutritional_requirements(row, assoc_row, target_nutrients):
                            associated_foods.append(assoc_row['Food'])
                            assoc_calorie = assoc_row['Energy(kcal)']
                            combined_cal = assoc_calorie + calorie
                            cal.append(combined_cal)
                            break
                        else:
                            food_df = divide_by_serving_combo(row, assoc_row, target_nutrients)
                            if len(food_df) == 4:
                                combined_cal = food_df[3] + food_df[1]
                                cal.append(combined_cal)
                                associated_foods.append(food_df[2])
                                break
                            elif len(food_df) == 2 and check_nutritional_requirements(row, target_nutrients):
                                if not associated_foods:
                                    recommendations_with_associations.append([row['Food'], '', row['Carbon Footprint(kg CO2e)'], food_df[1]])
                                    break

        if associated_foods:
            combined_cal = sum(cal) / len(cal)
            recommendations_with_associations.append([food_item, ', '.join(associated_foods), carbon_footprint, combined_cal])
        elif '0' in associativity_values:
            if check_nutritional_requirements(row, target_nutrients):
                recommendations_with_associations.append([food_item, '', carbon_footprint, calorie])
            else:
                df = divide_by_serving(row)
                if check_nutritional_requirements(df, target_nutrients):
                    recommendations_with_associations.append([df['Food'], '', df['Carbon Footprint(kg CO2e)'], df['Energy(kcal)']])

    while len(recommendations_with_associations) < 7:
        recommendations_with_associations.extend(recommendations_with_associations[:7 - len(recommendations_with_associations)])

    return pd.DataFrame(recommendations_with_associations[:7], columns=['Food', 'Associations', 'Carbon Footprint(kg CO2e)', 'Energy(kcal)'])
