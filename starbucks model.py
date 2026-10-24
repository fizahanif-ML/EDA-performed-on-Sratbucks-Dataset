%%writefile starbucks_complete_pipeline.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ========================================== #
# 1. LOAD & CLEAN DATA                       #
# ========================================== #
print("--- Loading and Preprocessing Data ---")
df = pd.read_csv('starbucks.csv')
df.columns = df.columns.str.strip()

numeric_cols = ['Calories', 'Total Fat (g)', 'Trans Fat (g)', 'Saturated Fat (g)',
                'Sodium (mg)', 'Total Carbohydrates (g)', 'Cholesterol (mg)',
                'Dietary Fibre (g)', 'Sugars (g)', 'Protein (g)']

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

features = ['Total Fat (g)', 'Trans Fat (g)', 'Saturated Fat (g)', 'Sodium (mg)',
            'Total Carbohydrates (g)', 'Cholesterol (mg)', 'Dietary Fibre (g)',
            'Sugars (g)', 'Protein (g)']

# Impute missing values in features using the median
for col in features:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

# Define the target variable (high calorie: > median)
median_calories = df['Calories'].median()
df['High_Calorie'] = (df['Calories'] > median_calories).astype(int)
print(f"Data preprocessed successfully. Median Calories threshold: {median_calories}\n")

# ========================================== #
# 2. EXPLORATORY DATA ANALYSIS (EDA)        #
# ========================================== #
print("--- Generating Visualizations ---")
# Heatmap of correlation matrix
plt.figure(figsize=(8, 6))
corr_cols = [c for c in ['Calories', 'Total Fat (g)', 'Sugars (g)', 'Protein (g)', 'Sodium (mg)'] if c in df.columns]
sns.heatmap(df[corr_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Nutritional Content')
plt.tight_layout()
plt.savefig('nutritional_correlations.png')
plt.close()

# Pie Chart of Calories by Category
category_calories = df.groupby('Beverage_category')['Calories'].sum()
plt.figure(figsize=(8, 8))
plt.pie(category_calories, labels=category_calories.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Proportion of Total Calories by Beverage Category')
plt.tight_layout()
plt.savefig('calories_pie_chart.png')
plt.close()

print("Plots saved as 'nutritional_correlations.png' and 'calories_pie_chart.png'.\n")

# ========================================== #
# 3. TRAIN DECISION TREE CLASSIFIER          #
# ========================================== #
print("--- Training Model ---")
X = df[features]
y = df['High_Calorie']

# Train (70%), Test (20%), Validation (10%) splits
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)

clf = DecisionTreeClassifier(random_state=42, max_depth=4)
clf.fit(X_train, y_train)

val_acc = accuracy_score(y_val, clf.predict(X_val))
test_acc = accuracy_score(y_test, clf.predict(X_test))

print(f"Validation Accuracy: {val_acc:.2%}")
print(f"Test Accuracy: {test_acc:.2%}\n")

# ========================================== #
# 4. SAVE AND EXPORT THE MODEL               #
# ========================================== #
model_filename = 'starbucks_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(clf, file)
print(f"Model serialized and saved successfully as '{model_filename}'!\n")

# ========================================== #
# 5. TEST LOADED MODEL WITH NEW PREDICTIONS   #
# ========================================== #
print("--- Verifying Saved Model with Inference Sample ---")
with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)

sample_drink = pd.DataFrame([{
    'Total Fat (g)': 3.5,
    'Trans Fat (g)': 0.2,
    'Saturated Fat (g)': 0.1,
    'Sodium (mg)': 15.0,
    'Total Carbohydrates (g)': 40.0,
    'Cholesterol (mg)': 15.0,
    'Dietary Fibre (g)': 0.0,
    'Sugars (g)': 35.0,
    'Protein (g)': 12.0
}])

pred = loaded_model.predict(sample_drink)[0]
label_map = {0: 'No (<=185 Calories)', 1: 'Yes (>185 Calories)'}
print(f"Prediction on mock drink: Is it high-calorie? -> {label_map[pred]}")
