import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


data = pd.read_csv("C:/Users/priya/Downloads/netflix/netflix_user_behavior_dataset.csv")

print(data.columns)


data["age"].fillna(data["age"].mean(), inplace=True)
data["avg_watch_time_minutes"].fillna(data["avg_watch_time_minutes"].mean(), inplace=True)


#BAR GRAPH

top_users = data.groupby("age")["avg_watch_time_minutes"].mean()

plt.figure(figsize=(10,6))
top_users.plot(kind='bar', color='skyblue')
plt.title("Average Watch Time by Age")
plt.xlabel("Age")
plt.ylabel("Watch Time")
plt.show()


#PIE CHART

device_data = data["devices_used"].value_counts()

plt.figure(figsize=(6,6))
plt.pie(device_data, labels=device_data.index, autopct='%1.1f%%')
plt.title("Device Usage")
plt.show()


# HISTOGRAM

plt.figure(figsize=(10,6))
sns.histplot(data["avg_watch_time_minutes"], kde=True, color='green')
plt.title("Watch Time Distribution")
plt.show()


# SCATTER + REGRESSION 

plt.figure(figsize=(10,6))

sns.regplot(
    x=data["age"],
    y=data["avg_watch_time_minutes"],
    scatter_kws={'alpha':0.3},   
    line_kws={'color':'red'}
)

plt.title("Age vs Watch Time (Regression Line)")
plt.xlabel("Age")
plt.ylabel("Watch Time")
plt.show()


#HEATMAP 

plt.figure(figsize=(10,6))

sns.heatmap(
    data.corr(numeric_only=True),
    annot=True,
    fmt=".2f",          
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap")
plt.show()


#  BOXPLOT

plt.figure(figsize=(10,6))
sns.boxplot(data=data.select_dtypes(include='number'))
plt.title("Outlier Detection")
plt.show()


# BAR GRAPH

subscription = data["subscription_type"].value_counts()

plt.figure(figsize=(8,5))
subscription.plot(kind='bar', color='orange')
plt.title("Subscription Types")
plt.show()


# LINEAR REGRESSION 

X = data[["age"]]
y = data["avg_watch_time_minutes"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# MODEL VISUALIZATION

plt.figure(figsize=(8,5))

plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red')

plt.title("Actual vs Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()


# METRICS

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
