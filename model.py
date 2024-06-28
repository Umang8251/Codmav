import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding, Dense, concatenate, Input
from tensorflow.keras.utils import to_categorical
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np

# Load the dataset
file_path = 'nutrition_cf.csv'
data = pd.read_csv(file_path)

# Define hyperparameters
num_ingredients = 1000  # Adjust based on your data
max_ingredients = 10  # Adjust based on your data
embedding_dim = 128
hidden_size = 256
num_recipes = len(data)  # Number of recipes in the dataset
num_nutrients = 4  # Energy, Proteins, Carbohydrates, Fats

# Preprocess categorical data
allergy_encoder = LabelEncoder()
data['Allergy'] = allergy_encoder.fit_transform(data['Allergy'])

# Extract ingredients and create a simple integer encoding for demonstration
ingredient_list = list(set(','.join(data['Ingredients'].str.split('; ')).split(', ')))
ingredient_encoder = {ingredient: idx for idx, ingredient in enumerate(ingredient_list, start=1)}
data['Ingredients'] = data['Ingredients'].apply(lambda x: [ingredient_encoder[ing] for ing in x.split('; ')])

# Pad ingredients to max_ingredients
data['Ingredients'] = data['Ingredients'].apply(lambda x: x[:max_ingredients] + [0]*(max_ingredients - len(x)))

# Extract target variables
nutrients = data[['Energy(kcal)', 'Proteins', 'Carbohydrates', 'Fats']].values
carbon_footprint = data['Carbon Footprint(kg/CO2)'].values

# Split data into training and test sets
train_data, test_data, train_nutrients, test_nutrients, train_carbon, test_carbon = train_test_split(
    data, nutrients, carbon_footprint, test_size=0.2, random_state=42
)

# Define model inputs
allergy_input = Input(shape=(1,), dtype=tf.int32, name="allergy_input")
ingredient_input = Input(shape=(max_ingredients,), dtype=tf.int32, name="ingredient_input")

# Define embedding layers
ingredient_embedding = Embedding(num_ingredients, embedding_dim)(ingredient_input)

# Combine inputs
combined_inputs = concatenate([ingredient_embedding, tf.cast(allergy_input, tf.float32)])

# Define hidden layers
hidden1 = Dense(hidden_size, activation="relu")(combined_inputs)
hidden2 = Dense(hidden_size, activation="relu")(hidden1)

# Output layers
recipe_output = Dense(num_recipes, activation="softmax", name="recipe_output")(hidden2)
carbon_output = Dense(1, activation="linear", name="carbon_output")(hidden2)

# Compile model
model = keras.Model(inputs=[allergy_input, ingredient_input], outputs=[recipe_output, carbon_output])
model.compile(loss={
    "recipe_output": tf.keras.losses.SparseCategoricalCrossentropy(),
    "carbon_output": "mse"
}, optimizer="adam", loss_weights=[0.8, 0.2])

# Prepare inputs
train_ingredients = np.array(train_data['Ingredients'].tolist())
train_allergies = np.array(train_data['Allergy'])

# Train the model
model.fit([train_allergies, train_ingredients], [train_data.index, train_carbon], epochs=10, batch_size=32)

# Make predictions on test data
test_ingredients = np.array(test_data['Ingredients'].tolist())
test_allergies = np.array(test_data['Allergy'])
predicted_recipe, estimated_carbon_footprint = model.predict([test_allergies, test_ingredients])

# Get the most likely recipe
recommended_recipe_id = tf.argmax(predicted_recipe[0]).numpy()
print(f"Recommended Recipe ID: {recommended_recipe_id}")
print(f"Estimated Carbon Footprint: {estimated_carbon_footprint[0]}")
