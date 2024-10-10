import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# 1. Load the data
# Replace this with your actual dataset
data = pd.read_csv('inputs/sample_customer_data_for_exam.csv')  # Load your dataset here

# 2. Prepare the data for modeling
# Assuming 'purchase_amount' is the target variable
X = data.drop('purchase_amount', axis=1)
y = data['purchase_amount']

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(exclude=['object']).columns.tolist()

# Create a preprocessor for handling missing values and encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numerical_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# 3. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Implement a regression model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Fit the model
model.fit(X_train, y_train)

# 5. Evaluate the model's performance
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# 6. Identify the top three features contributing to predicting the purchase amount
# Get feature importances from the Random Forest model
feature_importances = model.named_steps['regressor'].feature_importances_

# Get the names of the transformed features
ohe = model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
categorical_feature_names = ohe.get_feature_names_out(categorical_features)
all_feature_names = np.concatenate([numerical_features, categorical_feature_names])

# Combine feature names with their importance
feature_importance_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': feature_importances})
top_features = feature_importance_df.sort_values(by='Importance', ascending=False).head(3)

print("Top 3 features contributing to predicting purchase amount:")
print(top_features)
