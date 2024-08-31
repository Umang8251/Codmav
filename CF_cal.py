import csv
import re

def extract_ingredients(ingredients_string):
    """
    Extracts ingredients and their weights from a string.

    Args:
        ingredients_string: A string containing ingredients and their weights.

    Returns:
        A list of tuples, where each tuple contains an ingredient and its weight.
    """
    ingredients_list = []
    for ingredient_pair in ingredients_string.split(';'):
        match = re.match(r'(.*)-(\d+) gms', ingredient_pair)
        if match:
            ingredient = match.group(1).strip()  # Extract the ingredient and strip whitespace
            weight = int(match.group(2))  # Extract the weight
            ingredients_list.append((ingredient, weight))
    return ingredients_list

# Read carbon footprint data from CF.csv
cf_dict = {}
with open('CF.csv', 'r') as cf_file:
    cf_reader = csv.reader(cf_file)
    next(cf_reader)  # Skip the header row
    for row in cf_reader:
        ingredient = row[0].strip()
        carbon_emission = float(row[1])
        cf_dict[ingredient] = carbon_emission

# Open the new CSV file to write the dish names and their corresponding carbon footprints
with open('Dish_Carbon_Footprint.csv', 'w', newline='') as output_file:
    writer = csv.writer(output_file)
    writer.writerow(['Dish', 'Total Carbon Footprint (kg CO2e)'])  # Write the header

    # Read FOOD.csv and calculate carbon footprint for each dish
    with open(' ICE.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            dish = row[0]
            ingredients_string = row[1]
            ingredients = extract_ingredients(ingredients_string)
            total_carbon_footprint = 0
            for ingredient, weight in ingredients:
                carbon_emission = cf_dict.get(ingredient, 0)  # Get carbon emission factor, default to 0 if not found
                ingredient_cf = weight * carbon_emission  # Calculate carbon footprint for the ingredient
                total_carbon_footprint += ingredient_cf
            writer.writerow([dish, total_carbon_footprint])  # Write the dish and its total carbon footprint to the new CSV file
