import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the data
# Replace this with your actual dataset
data = pd.read_csv('inputs/sample_customer_data_for_exam.csv')  # Load your dataset here

# 2. Prepare the data for classification
# Assuming 'promotion_usage' is the target variable and is binary (0 or 1)
X = data.drop('promotion_usage', axis=1)
y = data['promotion_usage']

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

# 4. Implement a classification model
# You can choose either RandomForestClassifier or LogisticRegression
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))  # Change to LogisticRegression() if desired
])

# Fit the model
model.fit(X_train, y_train)

# 5. Evaluate the model
y_pred = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# 6. Create a confusion matrix and interpret results
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# 7. Discuss the top three factors contributing to predicting promotional offer usage
if isinstance(model.named_steps['classifier'], RandomForestClassifier):
    feature_importances = model.named_steps['classifier'].feature_importances_
elif isinstance(model.named_steps['classifier'], LogisticRegression):
    feature_importances = np.abs(model.named_steps['classifier'].coef_[0])

# Get the names of the transformed features
ohe = model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
categorical_feature_names = ohe.get_feature_names_out(categorical_features)
all_feature_names = np.concatenate([numerical_features, categorical_feature_names])

# Combine feature names with their importance
feature_importance_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': feature_importances})
top_features = feature_importance_df.sort_values(by='Importance', ascending=False).head(3)

print("Top 3 features contributing to predicting promotional offer usage:")
print(top_features)
