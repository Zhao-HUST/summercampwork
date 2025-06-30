import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv(r'C:\Users\34165\Desktop\summercampwork\summercampwork\day1\train.csv')
df = data.copy()

print('Random 10 rows of data information:')
print(df.sample(10).to_csv(sep='\t', na_rep='nan'))

# Delete some features that are not useful for prediction
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
print('Data basic information:')
df.info()

# Check if there is any NaN in the dataset
print('Is there any NaN in the dataset: {}'.format(df.isnull().values.any()))
df.dropna(inplace=True)
print('Is there any NaN in the dataset: {}'.format(df.isnull().values.any()))

# Convert categorical data into numerical data using one - hot encoding
df = pd.get_dummies(df)

print('Random 10 rows of encoded data information:')
print(df.sample(10).to_csv(sep='\t', na_rep='nan'))

# Separate the features and labels
X = df.drop(columns=['Survived'])
y = df['Survived']

# Train - test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Build model
# SVM
svm_model = SVC()
svm_model.fit(X_train, y_train)

# KNN
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Predict and evaluate
svm_pred = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)

knn_pred = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)

rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

print('SVM accuracy:', svm_accuracy)
print('KNN accuracy:', knn_accuracy)
print('Random Forest accuracy:', rf_accuracy)