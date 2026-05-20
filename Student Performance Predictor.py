import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

data = {
    'Study_Hours': [1, 2, 3, 4, 5, 6, 7, 8, 2, 5, 6, 1, 9, 4, 7],
    'Attendance': [60, 65, 70, 75, 80, 85, 90, 95, 68, 82, 88, 55, 98, 72, 91],
    'Sleep_Hours': [5, 6, 5, 7, 6, 7, 8, 7, 5, 6, 8, 4, 8, 6, 7],
    'Marks': [35, 40, 50, 55, 65, 70, 80, 88, 45, 68, 75, 30, 95, 58, 84]
}
df = pd.DataFrame(data)

print("\nDataset:\n")
print(df)
X = df[['Study_Hours', 'Attendance', 'Sleep_Hours']]
y = df['Marks']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nActual Marks:")
print(y_test.values)

print("\nPredicted Marks:")
print(predictions)

# Accuracy Checking
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nMean Absolute Error:", mae)
print("R2 Score:", r2)

print("\nEnter Student Details for Prediction")

study = float(input("Study Hours: "))
attendance = float(input("Attendance Percentage: "))
sleep = float(input("Sleep Hours: "))

new_data = pd.DataFrame({
    'Study_Hours': [study],
    'Attendance': [attendance],
    'Sleep_Hours': [sleep]
})

predicted_marks = model.predict(new_data)
print("\nPredicted Marks:", round(predicted_marks[0], 2))

plt.scatter(df['Study_Hours'], df['Marks'])
plt.xlabel('Study Hours')
plt.ylabel('Marks')
plt.title('Study Hours vs Marks')
plt.show()

