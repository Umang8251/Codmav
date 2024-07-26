#app.py
import streamlit as st
import pandas as pd
import re
from recommender import *
import requests
#from recommender import load_data, filter_dataset, recommend_food, prepare_recommendations_with_associative_rules
from img_finder import get_images_links
from streamlit_lottie import st_lottie
st.set_page_config(page_title="Recommender System", page_icon=":tada:", layout="wide")

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
    
lottie_coding = load_lottieurl("https://lottie.host/0193aee4-27d6-4b30-ad1f-33ab0f07bde3/nkAKPlrlyd.json")


st.title(":rainbow[Personalized Food Recommender]")
file_path = 'nutrition_cf - Sheet4.csv'
nutrition_data = load_data(file_path)
with st.container():
    left_column, right_column = st.columns(2)
    with left_column:
        user_height = st.number_input('Enter your height (in m):',
                            format="%f", value=1.0
                            )
        gender = st.radio(
            "Select your gender",
            [":rainbow[Male]", ":rainbow[Female]"],
            horizontal=True,
            )
        user_allergies_input = st.multiselect("Select your allergies", ["No-Allergies", "Dairy", "Egg", "Gluten", "Nut", "Soy", "Fish", "Mushroom", "Peanut", "Seafood", "Pork", "Onion", "Citrus", "Caffeine", "Garlic"])
        if "No-Allergies" in user_allergies_input:
            allergies = ["no-allergies"]
        else:
            allergies = user_allergies_input
    

    with right_column:
        user_weight = st.number_input('Enter your weight (in kg)',
                            format="%f", value=1.0
                            )
        user_category = st.selectbox("Select your diet preference", ["veg", "eggetarian", "non-veg"],
                             index=None,
                             placeholder=""
                             )
        user_region_pattern = st.selectbox("Select your region", ["North", "South", "East", "West", "Continental"],
                                   index=None,
                                   placeholder="")
        
bmi = (user_weight)/pow(user_height,2)        
# Define associativity rules
associativity_rules = {
    #'PITA BREAD (WHEAT)': ['Hummus', 'TABBOULEH'],
    '1': ['2'],
    '3': ['2', '4', '8'],
    '5': ['4', '8'],
    '6': ['7'],
    '11': ['10','12'],
    '14': ['13'],
    '9': ['10','12']
    # Add more rules as needed
}

# Set target values based on BMI and gender
target_values = set_target_values(bmi, gender)

target_breakfast = target_values['breakfast']
target_lunch = target_values['lunch']
target_snacks = target_values['snacks']
target_dinner = target_values['dinner']
target_appetizers = target_values['appetizers']

if st.button("Generate Diet Plan", type="primary"):
    animation_placeholder = st.empty()
    with animation_placeholder:
        st_lottie(lottie_coding, height=150, key="coding")
    try:
        filtered_data = filter_dataset(nutrition_data, user_allergies_input, user_region_pattern, user_category)
        filtered_data_null = filter_dataset(nutrition_data, user_allergies_input, "Null", user_category)
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

    # Prepare recommendations with associative rules
    recommended_breakfast_with_assoc = get_weekly_plan(recommended_breakfast, associativity_rules, ['0', '1', '3', '5','9','14'], target_breakfast)
    recommended_lunch_with_assoc = get_weekly_plan(recommended_lunch, associativity_rules, ['0', '1', '3', '5','14'], target_lunch)
    recommended_appetizers_with_assoc = get_weekly_plan(recommended_appetizers, associativity_rules, ['0', '6'], target_appetizers)
    recommended_dinner_with_assoc = get_weekly_plan(recommended_dinner, associativity_rules, ['0', '1', '3', '5','14'], target_dinner)
    recommended_snacks_with_assoc = get_weekly_plan(recommended_snacks, associativity_rules, ['0', '6', '11'], target_snacks)

    def to_get_searchterm(df):
        values = []
        for index, row in df.iterrows():
            value = row['Food']
            values.append(value)
        return values

    def fetch_images_for_recommendations(search_terms):
        images = {}
        for term in search_terms:
            images[term] = get_images_links(term)
        return images

    search_terms_breakfast = to_get_searchterm(recommended_breakfast_with_assoc)
    search_terms_lunch = to_get_searchterm(recommended_lunch_with_assoc)
    search_terms_appetizers = to_get_searchterm(recommended_appetizers_with_assoc)
    search_terms_dinner = to_get_searchterm(recommended_dinner_with_assoc)
    search_terms_snacks = to_get_searchterm(recommended_snacks_with_assoc)

    images_breakfast = fetch_images_for_recommendations(search_terms_breakfast)
    images_lunch = fetch_images_for_recommendations(search_terms_lunch)
    images_appetizers = fetch_images_for_recommendations(search_terms_appetizers)
    images_dinner = fetch_images_for_recommendations(search_terms_dinner)
    images_snacks = fetch_images_for_recommendations(search_terms_snacks)

    def display_recommendations_with_images(recommendations, images, title):
    # Define column sizes as constants
        column_styles = {
            'Meal Type': 'width: 100px;',
            'Image': 'width: 100px;',
            'Food Dish': 'width: 200px;',
            'Nutrition Info': 'width: 300px;',
            'Calories': 'width: 100px;',
            'Carbon Footprint': 'width: 150px;'
        }
        
        # Create the table header with column styles
        table_header = f"""
        <table>
            <thead>
                <tr>
                    <th style="{column_styles['Meal Type']}">Meal Type</th>
                    <th style="{column_styles['Image']}">Image</th>
                    <th style="{column_styles['Food Dish']}">Main Dish</th>
                    <th style="{column_styles['Nutrition Info']}">Side Dish</th>
                    <th style="{column_styles['Calories']}">Calories</th>
                    <th style="{column_styles['Carbon Footprint']}">Carbon Footprint</th>
                </tr>
            </thead>
            <tbody>
        """

        table_data = []

        for index, row in recommendations.iterrows():
            food = row['Food']
            food_link = f'<a href="https://www.google.com" target="_blank">{food}</a>'  # Replace with actual link format
            carbon_footprint = row['Carbon Footprint(kg CO2e)']
            image_urls = images.get(food, ["Not_found_link"])
            calories = row['Energy(kcal)']
            img_html = f'<img src="{image_urls[0]}" width="100"/>'
            food_html = f'<strong>{food_link}</strong>'
            carbon_html = f'{carbon_footprint} kg CO2e'
            calories_html = f'{calories} kcal'
            nutrition_html = row.to_frame().transpose().drop(columns=['Food', 'Carbon Footprint(kg CO2e)', 'Energy(kcal)']).to_html(index=False, header=False)
            table_data.append(f'<tr><td>{title}</td><td>{img_html}</td><td>{food_html}</td><td>{nutrition_html}</td><td>{calories_html}</td><td>{carbon_html}</td></tr>')

        table_footer = '</tbody></table>'
        table_html = table_header + "".join(table_data) + table_footer
        st.markdown(table_html, unsafe_allow_html=True)







    # Organize the recommendations into daily meal plans
    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    def organize_weekly_plan(recommendations_with_assoc):
        daily_meals = {day: [] for day in days_of_week}
        for i, day in enumerate(days_of_week):
            daily_meals[day] = recommendations_with_assoc.iloc[i::7].reset_index(drop=True)
        return daily_meals

    weekly_breakfast_plan = organize_weekly_plan(recommended_breakfast_with_assoc)
    weekly_lunch_plan = organize_weekly_plan(recommended_lunch_with_assoc)
    weekly_appetizers_plan = organize_weekly_plan(recommended_appetizers_with_assoc)
    weekly_dinner_plan = organize_weekly_plan(recommended_dinner_with_assoc)
    weekly_snacks_plan = organize_weekly_plan(recommended_snacks_with_assoc)

    animation_placeholder.empty()
    st.balloons()
    # Display recommendations using Streamlit
    st.write('# Recommended Foods')

    for day in days_of_week:
        st.write(f'## {day}')
        display_recommendations_with_images(weekly_breakfast_plan[day], images_breakfast, 'Breakfast')
        display_recommendations_with_images(weekly_appetizers_plan[day], images_appetizers, 'Appetizers')
        display_recommendations_with_images(weekly_lunch_plan[day], images_lunch, 'Lunch')
        display_recommendations_with_images(weekly_snacks_plan[day], images_snacks, 'Snacks')
        display_recommendations_with_images(weekly_dinner_plan[day], images_dinner, 'Dinner')