import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


df = pd.read_csv('gaurav.csv')

print(df.head())
#dataset information
print(df.info())
# stastical information of data of first 5 
print(df.describe)

#check the shape of dataset 
print(f"Dataset Shape: {df.shape}")

#checking for missing values
print(df.isnull().sum())




# Assuming df is your DataFrame
x = df.drop('PurchaseStatus', axis=1)  # Features
y = df['PurchaseStatus']  # Target

# Scaling the data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Splitting the data
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(x_train, y_train)

# Print intercept and coefficients
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")

# Make predictions
y_pred = model.predict(x_test)
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Plot using only the first feature
plt.scatter(x_scaled[:, 0], y, label="Actual Data", color="blue")  # First feature of x
plt.plot(x_scaled[:, 0], model.predict(x_scaled), color="red", label="Regression Line")
plt.legend()
plt.show()



# visualising data using matplotlip and sea born 

plt.figure(figsize=(6,4))
sns.countplot(x='PurchaseStatus', data=df)
plt.title('Distribution of PurchaseStatus')
plt.show()

plt.figure(figsize=(10,6))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='Gender', data=df)
plt.title('Gender Distribution')
plt.show()

plt.figure(figsize=(10,6))
sns.histplot(df['AnnualIncome'], bins=30, kde=True)
plt.title('Annual Income Distribution')
plt.show()

plt.figure(figsize=(10,6))
sns.histplot(df['NumberOfPurchases'], bins=30, kde=True)
plt.title('Number of Purchases Distribution')
plt.show()

plt.figure(figsize=(10,6))
sns.countplot(x='ProductCategory', data=df)
plt.title('Product Category Distribution')
plt.show()
x = df.drop('PurchaseStatus', axis=1)
y = df['PurchaseStatus']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

