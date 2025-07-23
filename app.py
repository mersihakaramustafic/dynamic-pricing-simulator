import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pylab as plt
from xgboost import plot_importance

# 1. EXPLORATORY DATA ANALYSIS

# Import data

df = pd.read_csv('retail_dataset.csv')

print(df.shape)
print(df.head(5))
print(df.dtypes)
print(df.describe())

# Data Cleaning

# Check for missing values
print(df.isna().sum())

# Check for duplicated values
print(df.loc[df.duplicated])

# Convert Data Type
df['date'] = pd.to_datetime(df['date'])
df['promo'] = df['promo'].astype('category')
df['category'] = df['category'].astype('category')
print(df.dtypes)

# 2. FEATURE ENGINEERING

df['price_diff_from_competitor'] = df['price'] - df['competitor_price']
df['is_discounted'] = (df['promo'] == 'Yes').astype(int)

print(df.head())

# 3. MODEL DEVELOPMENT

# XGBClassifier

# Define features and target
features = ['price', 'promo', 'competitor_price', 'category', 'price_diff_from_competitor']
target = ['purchased']

# Split the data into training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Train the model
model = XGBClassifier(n_estimators=100, enable_categorical=True, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train.values.ravel())

# Make predictions
y_pred = model.predict(X_test)

# Evaluation of the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Calculate Feature Importance
# Plot feature importance
plot_importance(model, max_num_features=10, importance_type='gain', xlabel='Importance (Gain)', title='XGBoost Feature Importance')
plt.show()