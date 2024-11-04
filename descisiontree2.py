import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


df = pd.read_csv(r"C:\Users\aswin\OneDrive\Desktop\data2.csv")

df.info()
df.describe()

le = LabelEncoder()
for column in df.columns:
    df[column] = le.fit_transform(df[column])

# Define features and target variable
X = df[['age', 'income', 'student', 'credit_rating']]  # Features
y = df['buys_computer']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree Classifier
dtc = DecisionTreeClassifier(criterion="entropy", random_state=42)
dtc.fit(X_train, y_train)



# After making predictions
y_pred = dtc.predict(X_test)

# Print evaluation metrics
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))
print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred))

plt.figure(figsize=(12, 8))
plot_tree(dtc, filled=True, feature_names=X.columns.tolist(), class_names=['No', 'Yes'], rounded=True)
plt.title('Decision Tree Visualization')
plt.show()
