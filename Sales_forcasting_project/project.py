import pandas as pd

df = pd.read_csv("train.csv")

print(df.head())
print(df.columns)
print(df.shape)
# Keep only required columns
df = df[['Order Date', 'Sales']]

print(df.head())
print(df.columns)
print(df.shape)
# Convert Order Date to datetime format
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
# Sort data by date
df = df.sort_values('Order Date')

print(df.head())
print(df.dtypes)
# Group by date and sum sales
df = df.groupby('Order Date')['Sales'].sum().reset_index()

print(df.head())
print(df.shape)
# Create time-based features
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Day'] = df['Order Date'].dt.day
df['DayOfWeek'] = df['Order Date'].dt.dayofweek

print(df.head())
print(df.columns)
# Define features and target
X = df[['Year', 'Month', 'Day', 'DayOfWeek']]
y = df['Sales']

print(X.head())
print(y.head())
from sklearn.model_selection import train_test_split

# Split data (no shuffle because time-series)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print("Training size:", X_train.shape)
print("Testing size:", X_test.shape)
from sklearn.linear_model import LinearRegression

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

print("Model trained successfully!")
# Predict on test data
predictions = model.predict(X_test)

print(predictions[:5])
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error:", mae)
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(y_test.values, label="Actual Sales")
plt.plot(predictions, label="Predicted Sales")
plt.legend()
plt.title("Actual vs Predicted Sales")
plt.show()