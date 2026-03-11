import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

# load dataset
df = pd.read_csv('diabetes_heart_factors.csv')

print("Original dataset shape:", df.shape)
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# data cleaning
df_clean = df.dropna(subset=['DIABETES', 'CHD'])

print("\nDataset shape after removing missing targets:", df_clean.shape)
print("\nRemaining missing values:")
print(df_clean.isnull().sum())

# descriptive statistics table
print("\n DESCRIPTIVE STATISTICS ")
desc_vars = ['DIABETES', 'CHD', 'OBESITY', 'CSMOKING', 'LPA', 'BINGE', 
             'median_household_income', 'median_age', 'pct_bachelors', 'pct_uninsured']
desc_stats = df_clean[desc_vars].describe().T
desc_stats['median'] = df_clean[desc_vars].median()
desc_stats = desc_stats[['mean', 'median', 'std', 'min', 'max']]
print(desc_stats)
desc_stats.to_csv('descriptive_statistics.csv')
print("\nDescriptive statistics saved to 'descriptive_statistics.csv'")

# select relevant features for analysis
features = ['BINGE', 'CSMOKING', 'LPA', 'OBESITY', 'median_household_income', 
            'median_age', 'pct_uninsured', 'pct_bachelors', 'pct_black']

# correlation analysis
correlation_data = df_clean[features + ['DIABETES', 'CHD']].copy()

# create correlation heatmap
plt.figure(figsize=(12, 10))
corr_matrix = correlation_data.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Matrix: Health and Demographic Factors')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300)
plt.close()

print("\nCorrelation with DIABETES:")
print(corr_matrix['DIABETES'].sort_values(ascending=False))
print("\nCorrelation with CHD:")
print(corr_matrix['CHD'].sort_values(ascending=False))

# save correlation results to csv
corr_results = pd.DataFrame({
    'Feature': corr_matrix.index,
    'Correlation_with_Diabetes': corr_matrix['DIABETES'],
    'Correlation_with_CHD': corr_matrix['CHD']
})
corr_results.to_csv('correlation_results.csv', index=False)
print("\nCorrelation results saved to 'correlation_results.csv'")

# hypothesis testing for key correlations
print("\n HYPOTHESIS TESTING ")
print("Testing significance of correlations (p-value < 0.05 indicates significant correlation)")

test_features = ['OBESITY', 'median_household_income', 'LPA', 'median_age', 'pct_bachelors']
hypothesis_results = []

for feature in test_features:
    # test for diabetes
    corr_diabetes, p_diabetes = stats.pearsonr(df_clean[feature], df_clean['DIABETES'])
    # test for CHD
    corr_chd, p_chd = stats.pearsonr(df_clean[feature], df_clean['CHD'])
    
    hypothesis_results.append({
        'Feature': feature,
        'Diabetes_Correlation': corr_diabetes,
        'Diabetes_p_value': p_diabetes,
        'Diabetes_Significant': 'Yes' if p_diabetes < 0.05 else 'No',
        'CHD_Correlation': corr_chd,
        'CHD_p_value': p_chd,
        'CHD_Significant': 'Yes' if p_chd < 0.05 else 'No'
    })
    
    print(f"\n{feature}:")
    print(f"  Diabetes: r={corr_diabetes:.4f}, p={p_diabetes:.4e} ({'significant' if p_diabetes < 0.05 else 'not significant'})")
    print(f"  CHD: r={corr_chd:.4f}, p={p_chd:.4e} ({'significant' if p_chd < 0.05 else 'not significant'})")

hypothesis_df = pd.DataFrame(hypothesis_results)
hypothesis_df.to_csv('hypothesis_testing_results.csv', index=False)
print("\nHypothesis testing results saved to 'hypothesis_testing_results.csv'")

# descriptive analysis visualizations
# 1. distribution of target variables (histogram)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(df_clean['DIABETES'], bins=30, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Diabetes Prevalence (%)')
axes[0].set_ylabel('Number of Counties')
axes[0].set_title('Distribution of Diabetes Prevalence')
axes[0].axvline(df_clean['DIABETES'].mean(), color='red', linestyle='--', label=f'Mean: {df_clean["DIABETES"].mean():.2f}%')
axes[0].legend()

axes[1].hist(df_clean['CHD'], bins=30, edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Heart Disease Prevalence (%)')
axes[1].set_ylabel('Number of Counties')
axes[1].set_title('Distribution of Heart Disease Prevalence')
axes[1].axvline(df_clean['CHD'].mean(), color='red', linestyle='--', label=f'Mean: {df_clean["CHD"].mean():.2f}%')
axes[1].legend()

plt.tight_layout()
plt.savefig('disease_distribution.png', dpi=300)
plt.close()

# 2. boxplot comparing key health factors
health_factors = ['OBESITY', 'CSMOKING', 'LPA', 'BINGE']
fig, ax = plt.subplots(figsize=(10, 6))
df_clean[health_factors].boxplot(ax=ax)
ax.set_ylabel('Prevalence (%)')
ax.set_title('Distribution of Health Risk Factors Across Counties')
ax.set_xticklabels(['Obesity', 'Smoking', 'Physical\nInactivity', 'Binge\nDrinking'])
plt.tight_layout()
plt.savefig('health_factors_boxplot.png', dpi=300)
plt.close()

# 3. income vs education scatter with disease overlay
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

scatter1 = axes[0].scatter(df_clean['median_household_income']/1000, 
                           df_clean['pct_bachelors'], 
                           c=df_clean['DIABETES'], 
                           cmap='YlOrRd', alpha=0.6, s=20)
axes[0].set_xlabel('Median Household Income ($1000s)')
axes[0].set_ylabel('Bachelor Degree Rate (%)')
axes[0].set_title('Income vs Education (colored by Diabetes Rate)')
cbar1 = plt.colorbar(scatter1, ax=axes[0])
cbar1.set_label('Diabetes (%)')

scatter2 = axes[1].scatter(df_clean['median_household_income']/1000, 
                           df_clean['pct_bachelors'], 
                           c=df_clean['CHD'], 
                           cmap='YlOrRd', alpha=0.6, s=20)
axes[1].set_xlabel('Median Household Income ($1000s)')
axes[1].set_ylabel('Bachelor Degree Rate (%)')
axes[1].set_title('Income vs Education (colored by Heart Disease Rate)')
cbar2 = plt.colorbar(scatter2, ax=axes[1])
cbar2.set_label('Heart Disease (%)')

plt.tight_layout()
plt.savefig('income_education_disease.png', dpi=300)
plt.close()

# scatter plots for key relationships
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].scatter(df_clean['OBESITY'], df_clean['DIABETES'], alpha=0.5)
axes[0, 0].set_xlabel('Obesity Rate (%)')
axes[0, 0].set_ylabel('Diabetes Rate (%)')
axes[0, 0].set_title('Obesity vs Diabetes')

axes[0, 1].scatter(df_clean['median_household_income'], df_clean['CHD'], alpha=0.5)
axes[0, 1].set_xlabel('Median Household Income ($)')
axes[0, 1].set_ylabel('Heart Disease Rate (%)')
axes[0, 1].set_title('Income vs Heart Disease')

axes[1, 0].scatter(df_clean['LPA'], df_clean['DIABETES'], alpha=0.5)
axes[1, 0].set_xlabel('Physical Inactivity Rate (%)')
axes[1, 0].set_ylabel('Diabetes Rate (%)')
axes[1, 0].set_title('Physical Inactivity vs Diabetes')

axes[1, 1].scatter(df_clean['median_age'], df_clean['CHD'], alpha=0.5)
axes[1, 1].set_xlabel('Median Age')
axes[1, 1].set_ylabel('Heart Disease Rate (%)')
axes[1, 1].set_title('Age vs Heart Disease')

plt.tight_layout()
plt.savefig('scatter_plots.png', dpi=300)
plt.close()

# prepare data for modeling
X = df_clean[features].fillna(df_clean[features].median())
y_diabetes = df_clean['DIABETES']
y_chd = df_clean['CHD']

# diabetes prediction model
X_train, X_test, y_train, y_test = train_test_split(X, y_diabetes, test_size=0.2, random_state=42)

rf_diabetes = RandomForestRegressor(n_estimators=100, random_state=42)
rf_diabetes.fit(X_train, y_train)
y_pred_train = rf_diabetes.predict(X_train)
y_pred_test = rf_diabetes.predict(X_test)

print("\n DIABETES PREDICTION MODEL ")
print(f"Training Set - R² Score: {r2_score(y_train, y_pred_train):.4f}")
print(f"Training Set - RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.4f}")
print(f"Test Set - R² Score: {r2_score(y_test, y_pred_test):.4f}")
print(f"Test Set - RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")

# residual analysis for diabetes
residuals_diabetes = y_test - y_pred_test

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(y_pred_test, residuals_diabetes, alpha=0.5)
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_xlabel('Predicted Diabetes Rate (%)')
axes[0].set_ylabel('Residuals')
axes[0].set_title('Residual Plot: Diabetes Model')

axes[1].hist(residuals_diabetes, bins=30, edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Residuals')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of Residuals: Diabetes Model')

plt.tight_layout()
plt.savefig('diabetes_residual_analysis.png', dpi=300)
plt.close()

# feature importance for diabetes
diabetes_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_diabetes.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance for Diabetes:")
print(diabetes_importance)

# save feature importance to csv
diabetes_importance.to_csv('diabetes_feature_importance.csv', index=False)
print("\nDiabetes feature importance saved to 'diabetes_feature_importance.csv'")

# heart disease prediction model
X_train, X_test, y_train, y_test = train_test_split(X, y_chd, test_size=0.2, random_state=42)

rf_chd = RandomForestRegressor(n_estimators=100, random_state=42)
rf_chd.fit(X_train, y_train)
y_pred_train = rf_chd.predict(X_train)
y_pred_test = rf_chd.predict(X_test)

print("\n HEART DISEASE PREDICTION MODEL ")
print(f"Training Set - R² Score: {r2_score(y_train, y_pred_train):.4f}")
print(f"Training Set - RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.4f}")
print(f"Test Set - R² Score: {r2_score(y_test, y_pred_test):.4f}")
print(f"Test Set - RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")

# residual analysis for heart disease
residuals_chd = y_test - y_pred_test

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(y_pred_test, residuals_chd, alpha=0.5)
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_xlabel('Predicted Heart Disease Rate (%)')
axes[0].set_ylabel('Residuals')
axes[0].set_title('Residual Plot: Heart Disease Model')

axes[1].hist(residuals_chd, bins=30, edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Residuals')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of Residuals: Heart Disease Model')

plt.tight_layout()
plt.savefig('chd_residual_analysis.png', dpi=300)
plt.close()

# feature importance for heart disease
chd_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_chd.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance for Heart Disease:")
print(chd_importance)

# save feature importance to csv
chd_importance.to_csv('chd_feature_importance.csv', index=False)
print("\nHeart disease feature importance saved to 'chd_feature_importance.csv'")

# model performance comparison table
X_train_diabetes, X_test_diabetes, y_train_diabetes, y_test_diabetes = train_test_split(X, y_diabetes, test_size=0.2, random_state=42)
X_train_chd, X_test_chd, y_train_chd, y_test_chd = train_test_split(X, y_chd, test_size=0.2, random_state=42)

model_performance = pd.DataFrame({
    'Model': ['Diabetes', 'Heart Disease'],
    'Train_R2': [r2_score(y_train_diabetes, rf_diabetes.predict(X_train_diabetes)),
                 r2_score(y_train_chd, rf_chd.predict(X_train_chd))],
    'Test_R2': [r2_score(y_test_diabetes, rf_diabetes.predict(X_test_diabetes)),
                r2_score(y_test_chd, rf_chd.predict(X_test_chd))],
    'Train_RMSE': [np.sqrt(mean_squared_error(y_train_diabetes, rf_diabetes.predict(X_train_diabetes))),
                   np.sqrt(mean_squared_error(y_train_chd, rf_chd.predict(X_train_chd)))],
    'Test_RMSE': [np.sqrt(mean_squared_error(y_test_diabetes, rf_diabetes.predict(X_test_diabetes))),
                  np.sqrt(mean_squared_error(y_test_chd, rf_chd.predict(X_test_chd)))]
})
model_performance.to_csv('model_performance_comparison.csv', index=False)
print("\nModel performance comparison saved to 'model_performance_comparison.csv'")

# compare feature importance between diabetes and heart disease
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].barh(diabetes_importance['feature'], diabetes_importance['importance'])
axes[0].set_xlabel('Importance')
axes[0].set_title('Feature Importance: Diabetes')
axes[0].invert_yaxis()

axes[1].barh(chd_importance['feature'], chd_importance['importance'])
axes[1].set_xlabel('Importance')
axes[1].set_title('Feature Importance: Heart Disease')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('feature_importance_comparison.png', dpi=300)
plt.close()

# summary statistics
print("\n SUMMARY STATISTICS ")
print("\nDiabetes prevalence by state (top 10):")
state_diabetes = df_clean.groupby('stateabbr')['DIABETES'].mean().sort_values(ascending=False).head(10)
print(state_diabetes)

print("\nHeart disease prevalence by state (top 10):")
state_chd = df_clean.groupby('stateabbr')['CHD'].mean().sort_values(ascending=False).head(10)
print(state_chd)

# save state rankings to csv
state_diabetes_full = df_clean.groupby('stateabbr')['DIABETES'].mean().sort_values(ascending=False)
state_chd_full = df_clean.groupby('stateabbr')['CHD'].mean().sort_values(ascending=False)

state_rankings = pd.DataFrame({
    'State': state_diabetes_full.index,
    'Diabetes_Rate': state_diabetes_full.values,
    'Diabetes_Rank': range(1, len(state_diabetes_full) + 1)
})
state_rankings['CHD_Rate'] = state_rankings['State'].map(state_chd_full)
state_rankings['CHD_Rank'] = state_rankings['State'].map(
    pd.Series(range(1, len(state_chd_full) + 1), index=state_chd_full.index)
)
state_rankings.to_csv('state_disease_rankings.csv', index=False)
print("\nState disease rankings saved to 'state_disease_rankings.csv'")

# save cleaned dataset
df_clean.to_csv('cleaned_diabetes_heart_data.csv', index=False)
print("\nCleaned dataset saved to 'cleaned_diabetes_heart_data.csv'")
print(f"Final dataset shape: {df_clean.shape}")
