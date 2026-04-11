# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_absolute_error

# 2. STYLE 

plt.style.use('ggplot')
sns.set_palette('Set2')


# 3. LOAD DATA

df = pd.read_excel(r"C:\Users\dell\OneDrive\Desktop\Numpy\crime_data.xlsx")

print("Dataset Preview:")
print(df.head())
print(df.shape)


# 4. DATA CLEANING 

df['ARREST_DATE'] = pd.to_datetime(df['ARREST_DATE'], errors='coerce')

df = df.drop_duplicates()
df = df.dropna(subset=['OFNS_DESC', 'LAW_CAT_CD', 'PERP_SEX', 'AGE_GROUP'])

df = df[df['PERP_SEX'].isin(['M', 'F'])]
df = df[df['AGE_GROUP'] != 'UNKNOWN']

df['LAW_CAT_CD'] = df['LAW_CAT_CD'].astype(str)
df['PERP_SEX'] = df['PERP_SEX'].astype(str)
df['AGE_GROUP'] = df['AGE_GROUP'].astype(str)


# 5. FEATURE ENGINEERING

df['year'] = df['ARREST_DATE'].dt.year
df['month'] = df['ARREST_DATE'].dt.month
df['day'] = df['ARREST_DATE'].dt.day_name()

print("Cleaned Data Shape:", df.shape)


# 6. VISUALIZATIONS 🎨

# Top Crime Types
plt.figure()
df['OFNS_DESC'].value_counts().head(10).plot(
    kind='bar', color=sns.color_palette('viridis')
)
plt.title("Top 10 Crime Types")
plt.xticks(rotation=45)
plt.show()

# Donut Chart
counts = df['LAW_CAT_CD'].value_counts()
colors = sns.color_palette('pastel')

plt.figure()
wedges, _, _ = plt.pie(
    counts,
    autopct='%1.1f%%',
    colors=colors,
    wedgeprops=dict(width=0.4)
)
plt.legend(wedges, counts.index, loc="center left", bbox_to_anchor=(1, 0.5))
plt.title("Crime Category (Donut Chart)")
plt.show()

# Heatmap
cross_tab = pd.crosstab(df['AGE_GROUP'], df['LAW_CAT_CD'])
plt.figure(figsize=(8,5))
sns.heatmap(cross_tab, annot=True, cmap='YlGnBu')
plt.title("Age vs Crime Category")
plt.show()

# Gender vs Crime
plt.figure()
sns.countplot(data=df, x='PERP_SEX', hue='LAW_CAT_CD')
plt.title("Crime Category by Gender")
plt.show()

# Monthly Trend
monthly = df['month'].value_counts().sort_index()
plt.figure()
monthly.plot(kind='bar', color='orange')
plt.title("Crimes by Month")
plt.show()


# 7. BOXPLOT

plt.figure()
sns.boxplot(x='LAW_CAT_CD', y='year', data=df)
plt.title("Crime Category vs Year (Boxplot)")
plt.show()


# 8. ENCODING 

le_crime = LabelEncoder()
le_sex = LabelEncoder()
le_age = LabelEncoder()

df['crime_encoded'] = le_crime.fit_transform(df['LAW_CAT_CD'])
df['sex_encoded'] = le_sex.fit_transform(df['PERP_SEX'])
df['age_encoded'] = le_age.fit_transform(df['AGE_GROUP'])


# 9. PAIRPLOT

sns.pairplot(df[['sex_encoded', 'age_encoded', 'crime_encoded']])
plt.suptitle("Pairplot", y=1.02)
plt.show()


# 10. LOCATION SCATTER

if 'Latitude' in df.columns and 'Longitude' in df.columns:
    plt.figure(figsize=(8,6))
    plt.scatter(df['Longitude'], df['Latitude'], c='red', alpha=0.3)
    plt.title("Crime Locations")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()


# 11. PREPARE DATA

X = df[['sex_encoded', 'age_encoded']]
y = df['crime_encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 12. RANDOM FOREST

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred))


# 13. LINEAR REGRESSION

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

print("\nLinear Regression Results:")
print("R2 Score:", r2_score(y_test, lr_pred))
print("MAE:", mean_absolute_error(y_test, lr_pred))


# 14. MODEL COMPARISON

print("\nModel Comparison:")
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
print("Linear Regression R2:", r2_score(y_test, lr_pred))


# 15. REGPLOT (LINEAR REGRESSION)

plt.figure()
sns.regplot(
    x=y_test,
    y=lr_pred,
    scatter_kws={'color': 'green'},
    line_kws={'color': 'red'}
)
plt.title("Linear Regression with Best Fit Line")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()


# 16. SHOW PREDICTIONS 

print("\nRandom Forest Predictions:")
print(y_pred[:10])

print("\nActual vs Predicted:")
for i in range(10):
    print("Actual:", y_test.iloc[i], "Predicted:", y_pred[i])


