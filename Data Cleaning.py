import pandas as pd

# Load your dataset
# Replace 'your_dataset.csv' with the path to your actual dataset file
df = pd.read_csv('food[1].csv')

# Display number of rows before removing duplicates
print("Number of rows before removing duplicates:", len(df))

# Remove duplicates
df.drop_duplicates(inplace=True)

# Display number of rows after removing duplicates
print("Number of rows after removing duplicates:", len(df))

# If you want to save the cleaned dataset back to a CSV file
# df.to_csv('cleaned_dataset.csv', index=False)
