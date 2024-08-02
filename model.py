#correct weekly plan
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
import numpy as np
import re
from math import pow

# Load the dataset
file_path = 'nutrition_cf - Sheet4.csv'
nutrition_data = pd.read_csv(file_path)

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
        region_filter = df['Region'].apply(lambda x: True)
        #return df[allergy_condition & category_condition]

    return df[allergy_condition & region_filter & category_condition]

## Function to set target nutritional values based on BMI and gender
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


# User inputs for allergies, region, and category
user_allergies_input = input("Enter allergies separated by commas (e.g., mushroom, peanut): ")
user_region_pattern = input("Enter region pattern (regex) (e.g., North): ")
user_category = input("Enter category (e.g., veg, non-veg, eggetarian): ")
user_gender = input("Enter your gender: ")
user_height = float(input("Enter your height: "))
user_weight = int(input("Enter your weight: "))

'''user_allergies_input = 'no-allergies'
user_region_pattern = "continental"
user_category = "non-veg"
user_gender = "male"
user_height = 1.85
user_weight = 80'''

print("User preference\nAllergy : "+user_allergies_input+"\nRegion : "+user_region_pattern+"\nCategory : "+user_category+"\n")

# Parsing user allergies input into a list
user_allergies = [allergy.strip() for allergy in user_allergies_input.split(',')]
if user_allergies == ['']:
    user_allergies = ["no-allergies"]

# Define associativity rules
associativity_rules = {
    '1': ['2'],
    '3': ['2', '4', '8'],
    '5': ['4', '8'],
    '6': ['7'],
    '11': ['10','12'],
    '14': ['13'],
    '9': ['10','12']
    # Add more rules as needed
}

bmi = (user_weight)/pow(user_height,2)
target_values = set_target_values(bmi, user_gender)
#print("BMI:",bmi)

target_breakfast = target_values['breakfast']
target_lunch = target_values['lunch']
target_snacks = target_values['snacks']
target_dinner = target_values['dinner']
target_appetizers = target_values['appetizers']

# Function to check if combined nutritional requirements are met
def check_combined_nutritional_requirements(food1, food2, target_nutrients):
    combined_nutrients = food1[['Proteins', 'Carbohydrates', 'Fats', 'Fiber', 'Energy(kcal)']] + food2[['Proteins', 'Carbohydrates', 'Fats', 'Fiber', 'Energy(kcal)']]
    meets_requirements = all(combined_nutrients <= pd.Series(target_nutrients))
    if(combined_nutrients <= pd.Series(target_nutrients)).all():
        return meets_requirements

def check_nutritional_requirements(food, target_nutrients,):
    combined_nutrients = food[['Proteins', 'Carbohydrates', 'Fats', 'Fiber', 'Energy(kcal)']]
    meets_requirements = all(combined_nutrients <= pd.Series(target_nutrients))
    if(combined_nutrients <= pd.Series(target_nutrients)).all():
        return meets_requirements

#To recommend 27 nearest neighbours based on the extract features using 'cosine' metric and 'auto' algorithm
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
    return recommended_foods.head(num_recommendations)

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

    return []


# Function to print recommendations with combinations
def get_weekly_plan(recommended_foods, associative_rules, valid_associations, target_nutrients):
    assoc_food_present = []
    weekly_plan = []
    for index, row in recommended_foods.iterrows():
        food_item = row['Food']
        associativity = str(row['Associativity'])
        calorie=row['Energy(kcal)']

        # Split associativity values if they are combined (e.g., '2,8')
        associativity_values = [value.strip() for value in associativity.split(',')]
        associated_foods = []
        cal=[]
        for value in associativity_values:
            if value in valid_associations and value in associative_rules:
                if (value=='11' or value=='9' or value=='5' or value=='3' or value=='1'):
                    associated_food_items = nutrition_data[nutrition_data['Associativity'].isin(associative_rules[value])]  
                else:
                    associated_food_items = recommended_foods[recommended_foods['Associativity'].isin(associative_rules[value])]     
                if (value!='14'):
                    associated_food_items = associated_food_items[~associated_food_items['Food'].isin(assoc_food_present)]
                if not associated_food_items.empty:
                    for _, assoc_row in associated_food_items.iterrows():
                        if check_combined_nutritional_requirements(row, assoc_row, target_nutrients):
                            assoc_food_present.append(assoc_row['Food'])
                            #print("if")
                            associated_foods.append(assoc_row['Food'])  
                            assoc_calorie=assoc_row['Energy(kcal)']
                            calorie = row['Energy(kcal)']
                            combined_cal = assoc_calorie + calorie
                            cal.append(combined_cal)                    
                            break 
                        else:
                            food_df = divide_by_serving_combo(row,assoc_row,target_nutrients)
                            if len(food_df)==4 :
                                #print("Check combo serving and printed")  
                                assoc_food_present.append(assoc_row['Food']) 
                                calorie = food_df[1]                           
                                assoc_foods = food_df[2]
                                assoc_calorie = food_df[3]
                                combined_cal = assoc_calorie + calorie
                                cal.append(combined_cal)
                                associated_foods.append(assoc_foods)
                                break                                

        if associated_foods :
            #print(f"{food_item, cal} (associated with: {', '.join(associated_foods)})")
            weekly_plan.append(f"{food_item,cal} (associated with: {', '.join(associated_foods)})")
        elif '0' in associativity_values:
          if check_nutritional_requirements(row, target_nutrients):
            #print(f"{food_item, calorie}")
            weekly_plan.append(f"{food_item,calorie}")
          else:
            df = divide_by_serving(row)
            if check_nutritional_requirements(df, target_nutrients):
              item = df['Food']
              calorie = df['Energy(kcal)']
              #print(f"{item, calorie}")
              weekly_plan.append(f"{item,calorie}")

    while len(weekly_plan) < 7:
        weekly_plan.extend(weekly_plan[:7 - len(weekly_plan)])

    return weekly_plan[:7]
# Filtering the dataset based on user preferences
try:
    filtered_data = filter_dataset(nutrition_data, user_allergies, user_region_pattern, user_category)
    filtered_data_null = filter_dataset(nutrition_data, user_allergies, "Null", user_category)
    try:
        recommended_breakfast = recommend_food(filtered_data, 'Breakfast', target_breakfast)
    except:
        recommended_breakfast = recommend_food(filtered_data_null, 'Breakfast', target_breakfast)

    try:
        recommended_snacks = recommend_food(filtered_data, 'Snacks', target_snacks)
    except:
        recommended_snacks = recommend_food(filtered_data_null, 'Snacks', target_snacks)

    try:
        recommended_appetizers = recommend_food(filtered_data, 'Appetizer', target_appetizers)
    except:
        recommended_appetizers = recommend_food(filtered_data_null, 'Appetizer', target_appetizers)

finally:
    recommended_lunch = recommend_food(filtered_data, 'Lunch', target_lunch)
    recommended_dinner = recommend_food(filtered_data, 'Dinner', target_dinner)


'''# Recommend food items for each meal type
print("Recommended Breakfast :"+ str(target_breakfast['Energy(kcal)']))
print_recommendations_with_associative_rules(recommended_breakfast, associativity_rules, ['0', '1', '3', '5','9','14'], target_breakfast)

print("\nRecommended Appetizers for Lunch :" + str(target_appetizers['Energy(kcal)']))
print_recommendations_with_associative_rules(recommended_appetizers, associativity_rules, ['0', '6'], target_appetizers)

print("\nRecommended Lunch :"+ str(target_lunch['Energy(kcal)']))
print_recommendations_with_associative_rules(recommended_lunch, associativity_rules, ['0', '1', '3', '5','14'], target_lunch)

print("\nRecommended Dinner :"+ str(target_dinner['Energy(kcal)']))
print_recommendations_with_associative_rules(recommended_dinner, associativity_rules, ['0', '1', '3', '5','14'], target_dinner)

print("\nRecommended Snacks :"+ str(target_snacks['Energy(kcal)']))
print_recommendations_with_associative_rules(recommended_snacks, associativity_rules, ['0', '11'], target_snacks)'''

breakfast_plan = get_weekly_plan(recommended_breakfast, associativity_rules, ['0', '1', '3', '5','9','14'], target_breakfast)
snacks_plan = get_weekly_plan(recommended_snacks, associativity_rules, ['0', '11'], target_snacks)
appetizers_plan = get_weekly_plan(recommended_appetizers, associativity_rules, ['0', '6'], target_appetizers)
lunch_plan = get_weekly_plan(recommended_lunch, associativity_rules,['0', '1', '3', '5','14'], target_lunch)
dinner_plan = get_weekly_plan(recommended_dinner, associativity_rules,['0', '1', '3', '5','14'], target_dinner)

# Format into a weekly plan
weekly_plan = {
    'Monday': {'Breakfast': breakfast_plan[0], 'Lunch': lunch_plan[0], 'Snacks': snacks_plan[0], 'Dinner': dinner_plan[0], 'Appetizers': appetizers_plan[0]},
    'Tuesday': {'Breakfast': breakfast_plan[1], 'Lunch': lunch_plan[1], 'Snacks': snacks_plan[1], 'Dinner': dinner_plan[1], 'Appetizers': appetizers_plan[1]},
    'Wednesday': {'Breakfast': breakfast_plan[2], 'Lunch': lunch_plan[2], 'Snacks': snacks_plan[2], 'Dinner': dinner_plan[2], 'Appetizers': appetizers_plan[2]},
    'Thursday': {'Breakfast': breakfast_plan[3], 'Lunch': lunch_plan[3], 'Snacks': snacks_plan[3], 'Dinner': dinner_plan[3], 'Appetizers': appetizers_plan[3]},
    'Friday': {'Breakfast': breakfast_plan[4], 'Lunch': lunch_plan[4], 'Snacks': snacks_plan[4], 'Dinner': dinner_plan[4], 'Appetizers': appetizers_plan[4]},
    'Saturday': {'Breakfast': breakfast_plan[5], 'Lunch': lunch_plan[5], 'Snacks': snacks_plan[5], 'Dinner': dinner_plan[5], 'Appetizers': appetizers_plan[5]},
    'Sunday': {'Breakfast': breakfast_plan[6], 'Lunch': lunch_plan[6], 'Snacks': snacks_plan[6], 'Dinner': dinner_plan[6], 'Appetizers': appetizers_plan[6]},
}

# Print the weekly plan
for day, meals in weekly_plan.items():
    print(f"{day}:")
    for meal_type, meal in meals.items():
        print(f"  {meal_type}: {meal}")
    print()   