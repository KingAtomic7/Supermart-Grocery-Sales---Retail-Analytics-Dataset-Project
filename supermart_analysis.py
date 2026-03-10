"""
========================================================
  Supermart Grocery Sales - Retail Analytics Project
  Internship Project | Data Analyst & Data Scientist
========================================================
Tools: Python, ML, SQL, Excel
Domain: Data Analyst & Data Scientist
Difficulty: Intermediate
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
os.makedirs('charts', exist_ok=True)

# ===========================================
# STEP 1: LOAD DATASET
# ===========================================
print("=" * 50)
print("STEP 1: Loading Dataset")
print("=" * 50)

df = pd.read_csv('supermart_data.csv')
print(f"Shape: {df.shape}")
print(df.head())

# ===========================================
# STEP 2: DATA PREPROCESSING
# ===========================================
print("\n" + "=" * 50)
print("STEP 2: Data Preprocessing")
print("=" * 50)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Drop duplicates
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Convert Order Date to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Extract temporal features
df['Order Day'] = df['Order Date'].dt.day
df['Order Month'] = df['Order Date'].dt.month
df['Order Year'] = df['Order Date'].dt.year

print(f"\nCleaned shape: {df.shape}")
print(df.dtypes)

# ===========================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ===========================================
print("\n" + "=" * 50)
print("STEP 3: Exploratory Data Analysis")
print("=" * 50)

# --- 3.1 Sales by Category ---
sales_category = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
print("\nSales by Category:")
print(sales_category)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#2196F3','#4CAF50','#FF5722','#9C27B0','#FF9800','#00BCD4','#F44336']
sales_category.plot(kind='bar', ax=ax, color=colors, edgecolor='white')
ax.set_title('Total Sales by Category', fontsize=16, fontweight='bold')
ax.set_xlabel('Category'); ax.set_ylabel('Total Sales (INR)')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('charts/01_category_sales.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart saved: charts/01_category_sales.png")

# --- 3.2 Yearly Sales Pie ---
yearly_sales = df.groupby("year")["Sales"].sum()
fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(yearly_sales.values, labels=yearly_sales.index, autopct='%1.1f%%',
       colors=['#2196F3','#4CAF50','#FF9800','#F44336'], startangle=90,
       wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2))
ax.set_title('Sales Distribution by Year', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('charts/02_yearly_sales_pie.png', dpi=150, bbox_inches='tight')
plt.close()

# --- 3.3 Monthly Sales Trend ---
monthly_sales = df.groupby('month_no')['Sales'].sum()
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(monthly_sales.index, monthly_sales.values, marker='o', linewidth=2.5,
        color='#2196F3', markersize=8, markerfacecolor='white', markeredgewidth=2)
ax.fill_between(monthly_sales.index, monthly_sales.values, alpha=0.1, color='#2196F3')
ax.set_title('Monthly Sales Trend', fontsize=16, fontweight='bold')
ax.set_xlabel('Month'); ax.set_ylabel('Total Sales (INR)')
ax.set_xticks(range(1, 13))
ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
plt.tight_layout()
plt.savefig('charts/03_monthly_sales.png', dpi=150, bbox_inches='tight')
plt.close()

# --- 3.4 Top 5 Cities ---
city_sales = df.groupby('City')['Sales'].sum().sort_values(ascending=False).head(5)
print("\nTop 5 Cities by Sales:")
print(city_sales)
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(city_sales.index, city_sales.values,
       color=['#1565C0','#1976D2','#1E88E5','#2196F3','#42A5F5'], edgecolor='white')
ax.set_title('Top 5 Cities by Sales', fontsize=16, fontweight='bold')
ax.set_xlabel('City'); ax.set_ylabel('Total Sales (INR)')
plt.tight_layout()
plt.savefig('charts/04_top5_cities.png', dpi=150, bbox_inches='tight')
plt.close()

# --- 3.5 Correlation Heatmap ---
le = LabelEncoder()
df_enc = df.copy()
for col in ['Category', 'Sub Category', 'City', 'Region', 'State', 'Month']:
    df_enc[col] = le.fit_transform(df_enc[col])
corr_cols = ['Category','Sub Category','City','Region','Sales','Discount','Profit','month_no','year']
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df_enc[corr_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
ax.set_title('Correlation Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('charts/05_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# --- 3.6 Profit by Region ---
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df, x='Region', y='Profit', palette='Set2', ax=ax)
ax.set_title('Profit Distribution by Region', fontsize=16, fontweight='bold')
ax.set_xlabel('Region'); ax.set_ylabel('Profit (INR)')
plt.tight_layout()
plt.savefig('charts/06_profit_region.png', dpi=150, bbox_inches='tight')
plt.close()

# ===========================================
# STEP 4: FEATURE ENGINEERING & ML
# ===========================================
print("\n" + "=" * 50)
print("STEP 4: Machine Learning Model")
print("=" * 50)

features = df_enc[['Category','Sub Category','City','Region','month_no','Discount','year']]
target = df_enc['Sales']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_s, y_train)
y_pred_lr = lr.predict(X_test_s)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print(f"\nLinear Regression - MSE: {mse_lr:.2f}, R²: {r2_lr:.4f}")

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"Random Forest     - MSE: {mse_rf:.2f}, R²: {r2_rf:.4f}")

# Actual vs Predicted chart
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(y_test, y_pred_lr, alpha=0.3, color='#2196F3', s=10)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r-', lw=2)
axes[0].set_title(f'Linear Regression\nR² = {r2_lr:.4f}', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Actual Sales'); axes[0].set_ylabel('Predicted Sales')
axes[1].scatter(y_test, y_pred_rf, alpha=0.3, color='#4CAF50', s=10)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r-', lw=2)
axes[1].set_title(f'Random Forest\nR² = {r2_rf:.4f}', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Actual Sales'); axes[1].set_ylabel('Predicted Sales')
plt.suptitle('Actual vs Predicted Sales', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('charts/07_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()

# Feature Importance
feat_imp = pd.Series(rf.feature_importances_, index=features.columns).sort_values()
fig, ax = plt.subplots(figsize=(9, 5))
feat_imp.plot(kind='barh', ax=ax, color='#2196F3')
ax.set_title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig('charts/08_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# ===========================================
# STEP 5: SUMMARY STATISTICS
# ===========================================
print("\n" + "=" * 50)
print("STEP 5: Summary Statistics")
print("=" * 50)
print(f"Total Sales:    INR {df['Sales'].sum():,.0f}")
print(f"Total Profit:   INR {df['Profit'].sum():,.0f}")
print(f"Avg Discount:   {df['Discount'].mean():.2%}")
print(f"Total Orders:   {len(df):,}")
print(f"Date Range:     {df['Order Date'].min().date()} to {df['Order Date'].max().date()}")
print(f"\nTop Category:   {sales_category.index[0]} (INR {sales_category.iloc[0]:,.0f})")
print(f"Top City:       {city_sales.index[0]} (INR {city_sales.iloc[0]:,.0f})")
print("\nAll charts saved to /charts/ folder!")
print("Project complete.")
