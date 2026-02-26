import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE
from collections import Counter
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('TrafficTwoMonth.csv')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Time period: {df['Date'].min()} to {df['Date'].max()}")
print("\nFirst 5 rows:")
print(df.head())

# 1. DATA UNDERSTANDING AND PREPROCESSING

# Create combined datetime column for reference
df['DateTime'] = pd.to_datetime('2024-01-' + df['Date'].astype(str) + ' ' + df['Time'], format='%Y-%m-%d %I:%M:%S %p')

# Extract hour and minute from time string
df['Hour'] = df['DateTime'].dt.hour
df['Minute'] = df['DateTime'].dt.minute

# Convert day of week from string to numeric
day_mapping = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
    'Friday': 4, 'Saturday': 5, 'Sunday': 6
}
df['DayOfWeek'] = df['Day of the week'].map(day_mapping)

# Create binary feature to distinguish weekdays from weekends
df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

# Create cyclical features to capture the circular nature of time
df['Hour_sin'] = np.sin(2 * np.pi * df['Hour']/24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour']/24)

# Create rush hour indicators
df['MorningRush'] = ((df['Hour'] >= 7) & (df['Hour'] <= 9)).astype(int)
df['EveningRush'] = ((df['Hour'] >= 16) & (df['Hour'] <= 19)).astype(int)
df['RushHour'] = (df['MorningRush'] | df['EveningRush']).astype(int)
df['NightTime'] = ((df['Hour'] >= 22) | (df['Hour'] <= 5)).astype(int)

# Calculate total vehicles for analysis only (not for training)
df['TotalVehicles'] = df['CarCount'] + df['BikeCount'] + df['BusCount'] + df['TruckCount']

# Encode target variable in logical order
traffic_order = {
    'low': 0,
    'normal': 1,
    'high': 2,
    'heavy': 3
}
df['Traffic_encoded'] = df['Traffic Situation'].map(traffic_order)

le = LabelEncoder()
le.fit(df['Traffic Situation'])  

class_names = ['low', 'normal', 'high', 'heavy']
print("\nEncoding Mapping (by severity):")
for traffic_class, encoded_value in traffic_order.items():
    print(f"   '{traffic_class}' → {encoded_value}")

# 2. EXPLORATORY DATA ANALYSIS

print("\nEXPLORATORY DATA ANALYSIS")
#EDA IMAGE 1
fig1 = plt.figure(figsize=(16, 12))

# Plot 1: Traffic pattern throughout the day
ax1 = plt.subplot(2, 2, 1)
hourly_traffic = df.groupby('Hour')['Traffic_encoded'].mean() * 25
hourly_volume = df.groupby('Hour')['TotalVehicles'].mean()
ax1.bar(hourly_traffic.index, hourly_traffic.values, alpha=0.7, color='skyblue', label='Traffic Severity')
ax1_twin = ax1.twinx()
ax1_twin.plot(hourly_volume.index, hourly_volume.values, color='red', marker='o', linewidth=2, label='Vehicle Count')
ax1.set_xlabel('Hour of Day')
ax1.set_ylabel('Traffic Severity (scaled)')
ax1_twin.set_ylabel('Average Vehicle Count')
ax1.set_title('Traffic Pattern by Hour\n(Higher severity = More vehicles)')
ax1.legend(loc='upper left')
ax1_twin.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Plot 2: Traffic severity by day of week
ax2 = plt.subplot(2, 2, 2)
dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
dow_traffic = df.groupby('DayOfWeek')['Traffic_encoded'].mean()
colors = ['red' if x < 5 else 'green' for x in range(7)]
bars = ax2.bar(dow_names, dow_traffic.values, color=colors, edgecolor='black')
ax2.set_xlabel('Day of Week')
ax2.set_ylabel('Average Traffic Severity\n(Higher = More Severe)')
ax2.set_title('Traffic Severity by Day')
ax2.grid(True, alpha=0.3)

# Plot 3: Vehicle composition for each traffic situation
ax3 = plt.subplot(2, 2, 3)
composition = df.groupby('Traffic Situation')[['CarCount', 'BikeCount', 'BusCount', 'TruckCount']].mean()
composition.plot(kind='bar', stacked=True, ax=ax3, colormap='viridis', edgecolor='black')
ax3.set_xlabel('Traffic Situation')
ax3.set_ylabel('Average Vehicle Count')
ax3.set_title('Vehicle Composition by Traffic Type')
ax3.legend(loc='upper right')
ax3.tick_params(axis='x', rotation=45)

# Plot 4: Distribution of traffic situations
ax4 = plt.subplot(2, 2, 4)
traffic_counts = df['Traffic Situation'].value_counts()
colors_4 = ['green', 'yellow', 'orange', 'red']
ax4.pie(traffic_counts.values, labels=traffic_counts.index, autopct='%1.1f%%', colors=colors_4, startangle=90)
ax4.set_title('Traffic Situation Distribution\n(Note: normal dominates)')

plt.tight_layout()
plt.savefig('01_eda_part1.png', dpi=150, bbox_inches='tight')
plt.show()

#EDA IMAGE 2
fig2 = plt.figure(figsize=(16, 12))

# Plot 1: Correlation matrix heatmap
ax1 = plt.subplot(2, 2, 1)
numeric_cols = ['Hour', 'DayOfWeek', 'IsWeekend', 'RushHour', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'TotalVehicles', 'Traffic_encoded']
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu', center=0, ax=ax1, square=True)
ax1.set_title('Feature Correlation Matrix\n(NOTE: Higher Traffic_encoded = More Severe Traffic)')

# Plot 2: Box plot showing vehicle distribution for each traffic type
ax2 = plt.subplot(2, 2, 2)
df.boxplot(column='TotalVehicles', by='Traffic Situation', ax=ax2, patch_artist=True)
ax2.set_xlabel('Traffic Situation')
ax2.set_ylabel('Total Vehicles')
ax2.set_title('Vehicle Distribution by Traffic Type')
plt.suptitle('')

# Plot 3: Impact of rush hour on traffic severity
ax3 = plt.subplot(2, 2, 3)
rush_data = df.groupby('RushHour')['Traffic_encoded'].mean()
bars = ax3.bar(['Non-Rush Hour', 'Rush Hour'], rush_data.values, color=['skyblue', 'red'], edgecolor='black')
ax3.set_ylabel('Average Traffic Severity\n(Higher = More Severe)')
ax3.set_title('Impact of Rush Hour on Traffic')

# Plot 4: Weekday vs weekend traffic comparison
ax4 = plt.subplot(2, 2, 4)
weekend_data = df.groupby('IsWeekend')['Traffic_encoded'].mean()
bars = ax4.bar(['Weekday', 'Weekend'], weekend_data.values, color=['blue', 'green'], edgecolor='black')
ax4.set_ylabel('Average Traffic Severity\n(Higher = More Severe)')
ax4.set_title('Weekday vs Weekend Traffic')

plt.tight_layout()
plt.savefig('01_eda_part2.png', dpi=150, bbox_inches='tight')
plt.show()

# 3. FEATURE PREPARATION
print("\nFEATURE PREPARATION")
# Sort data by datetime first
df = df.sort_values('DateTime').reset_index(drop=True)
# HANDLING OUTLIERS
print("\nHANDLING OUTLIERS")
Q1 = df['TotalVehicles'].quantile(0.25)
Q3 = df['TotalVehicles'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_before = len(df[(df['TotalVehicles'] < lower_bound) | (df['TotalVehicles'] > upper_bound)])
print(f"Outliers detected: {outliers_before} rows ({outliers_before/len(df)*100:.2f}%)")

# Cap outliers (replace extreme values with bounds)
df['TotalVehicles'] = df['TotalVehicles'].clip(lower_bound, upper_bound)
print("Outliers capped (not removed)")
df['TotalVehicles'] = df['CarCount'] + df['BikeCount'] + df['BusCount'] + df['TruckCount']

# FEATURE SET
feature_cols = ['Hour', 'Hour_sin', 'Hour_cos','DayOfWeek', 'IsWeekend','MorningRush', 'EveningRush', 'RushHour', 'NightTime',
    'TotalVehicles'
]

print(f"Total Features: {len(feature_cols)}")
print("\nFinal Features selected:")
for i, f in enumerate(feature_cols, 1):
    print(f"  {i}. {f}")

X = df[feature_cols]
y = df['Traffic_encoded']

# Time-based split with proper gap to prevent leakage
cutoff_date = df['DateTime'].iloc[int(0.8 * len(df))]
train_mask = df['DateTime'] < cutoff_date
test_mask = df['DateTime'] >= cutoff_date

X_train = X[train_mask]
X_test = X[test_mask]
y_train = y[train_mask]
y_test = y[test_mask]

print(f"\nTraining set: {X_train.shape[0]} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"Testing set: {X_test.shape[0]} samples ({X_test.shape[0]/len(df)*100:.1f}%)")
print(f"\nTraining period: {df['DateTime'].iloc[0]} to {df['DateTime'][train_mask].max()}")
print(f"Testing period: {df['DateTime'][test_mask].min()} to {df['DateTime'].iloc[-1]}")


# HANDLING CLASS IMBALANCE WITH SMOTE

print("\nHANDLING CLASS IMBALANCE WITH SMOTE")

# Check class distribution before SMOTE
print("\nClass distribution BEFORE SMOTE:")
print(f"Training set: {Counter(y_train)}")
print(f"Testing set: {Counter(y_test)}")

# Apply SMOTE to balance training data
smote = SMOTE(random_state=42, sampling_strategy='auto')
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("\nClass distribution AFTER SMOTE:")
print(f"Training set: {Counter(y_train_balanced)}")
print(f"All classes now have {Counter(y_train_balanced)[0]} samples")


# 4. MODEL TRAINING WITH BALANCED DATA

print("\nMODEL TRAINING WITH BALANCED DATA")

# Calculate class weights for reference
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
print(f"\nClass weights: {weight_dict}")

models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced')
}

results = {}
predictions = {}
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name} on balanced data...")
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    # Calculate metrics with focus on F1-score
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    f1_per_class = f1_score(y_test, y_pred, average=None)
    
    results[name] = {
        'accuracy': accuracy, 
        'precision': precision, 
        'recall': recall, 
        'f1': f1,
        'f1_per_class': f1_per_class
    }
    predictions[name] = y_pred
    trained_models[name] = model
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted F1-Score: {f1:.4f}")

# 5. XGBOOST MODEL TRAINING

print("\nXGBOOST MODEL TRAINING")

# Parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.1, 0.2],
    'subsample': [0.8, 1.0]
}

xgb_model = xgb.XGBClassifier(
    objective='multi:softprob', 
    num_class=4, 
    random_state=42, 
    use_label_encoder=False, 
    eval_metric='mlogloss'
)

stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Grid search with f1-weighted scoring
grid_search = GridSearchCV(
    xgb_model, 
    param_grid, 
    cv=stratified_kfold, 
    scoring='f1_weighted',
    n_jobs=-1, 
    verbose=0
)
grid_search.fit(X_train_balanced, y_train_balanced)

print("\nBest Parameters:")
for param, value in grid_search.best_params_.items():
    print(f"{param}: {value}")

best_xgb = grid_search.best_estimator_
y_pred_best = best_xgb.predict(X_test)
y_pred_best_proba = best_xgb.predict_proba(X_test)

best_accuracy = accuracy_score(y_test, y_pred_best)
best_precision = precision_score(y_test, y_pred_best, average='weighted')
best_recall = recall_score(y_test, y_pred_best, average='weighted')
best_f1 = f1_score(y_test, y_pred_best, average='weighted')
best_f1_per_class = f1_score(y_test, y_pred_best, average=None)
best_logloss = log_loss(y_test, y_pred_best_proba)

print(f"\nXGBoost Results:")
print(f"Weighted F1-Score: {best_f1:.4f}")
print(f"Accuracy: {best_accuracy:.4f}")

# 6. MODEL EVALUATION AND COMPARISON

print("\nMODEL EVALUATION")

comparison_df = pd.DataFrame({
    'Model': ['Decision Tree', 'XGBoost'],
    'Accuracy': [results['Decision Tree']['accuracy'], best_accuracy],
    'Precision': [results['Decision Tree']['precision'], best_precision],
    'Recall': [results['Decision Tree']['recall'], best_recall],
    'F1-Score': [results['Decision Tree']['f1'], best_f1]
})

print("\n Model Comparison:")
print(comparison_df.to_string(index=False))

# 7. RESULTS VISUALIZATION 

#RESULT IMAGE 1
fig3 = plt.figure(figsize=(16, 12))

# Plot 1: Model F1-Score comparison bar chart
ax1 = plt.subplot(2, 2, 1)
models_comp = ['Decision Tree', 'XGBoost']
f1_values = [results['Decision Tree']['f1'], best_f1]
colors_1 = ['lightgreen', 'coral']
bars = ax1.bar(models_comp, f1_values, color=colors_1, edgecolor='black')
ax1.set_ylabel('Weighted F1-Score')
ax1.set_title('Model F1-Score Comparison')
ax1.set_ylim(0, 1)
for bar, val in zip(bars, f1_values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', ha='center', va='bottom')

# Plot 2: Confusion matrix for Decision Tree
ax2 = plt.subplot(2, 2, 2)
cm_dt = confusion_matrix(y_test, predictions['Decision Tree'])
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Greens', ax=ax2, xticklabels=class_names, yticklabels=class_names)
ax2.set_title('Decision Tree - Confusion Matrix')
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')

# Plot 3: Confusion matrix for XGBoost
ax3 = plt.subplot(2, 2, 3)
cm_xgb = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Oranges', ax=ax3, xticklabels=class_names, yticklabels=class_names)
ax3.set_title('XGBoost - Confusion Matrix')
ax3.set_xlabel('Predicted')
ax3.set_ylabel('Actual')

# Plot 4: Side-by-side comparison of metrics
ax4 = plt.subplot(2, 2, 4)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
dt_values = [results['Decision Tree']['accuracy'], results['Decision Tree']['precision'], results['Decision Tree']['recall'], results['Decision Tree']['f1']]
xgb_values = [best_accuracy, best_precision, best_recall, best_f1]
x = np.arange(len(metrics))
width = 0.35
ax4.bar(x - width/2, dt_values, width, label='Decision Tree', color='lightgreen', edgecolor='black')
ax4.bar(x + width/2, xgb_values, width, label='XGBoost', color='coral', edgecolor='black')
ax4.set_xlabel('Metrics')
ax4.set_ylabel('Score')
ax4.set_title('Model Performance Comparison')
ax4.set_xticks(x)
ax4.set_xticklabels(metrics)
ax4.set_ylim(0, 1)
ax4.legend()

plt.tight_layout()
plt.savefig('02_results_part1.png', dpi=150, bbox_inches='tight')
plt.show()

#RESULT IMAGE 2
fig4 = plt.figure(figsize=(16, 12))

# Plot 1: Feature importance for Decision Tree
ax1 = plt.subplot(2, 2, 1)
dt_importance = pd.DataFrame({'feature': feature_cols, 'importance': trained_models['Decision Tree'].feature_importances_}).sort_values('importance', ascending=True)
ax1.barh(dt_importance['feature'][-10:], dt_importance['importance'][-10:], color='lightgreen', edgecolor='black')
ax1.set_xlabel('Importance')
ax1.set_title('Top 10 Features - Decision Tree')

# Plot 2: Feature importance for XGBoost
ax2 = plt.subplot(2, 2, 2)
xgb_importance = pd.DataFrame({'feature': feature_cols, 'importance': best_xgb.feature_importances_}).sort_values('importance', ascending=True)
ax2.barh(xgb_importance['feature'][-10:], xgb_importance['importance'][-10:], color='coral', edgecolor='black')
ax2.set_xlabel('Importance')
ax2.set_title('Top 10 Features - XGBoost')

# Plot 3: Learning curves for XGBoost (now using ax3, not ax4)
ax3 = plt.subplot(2, 2, 3)  # Changed from ax4 to ax3
train_sizes = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
train_scores = []
test_scores = []
for size in train_sizes:
    n_samples = int(len(X_train_balanced) * size)
    X_subset = X_train_balanced[:n_samples]
    y_subset = y_train_balanced[:n_samples]
    xgb_temp = xgb.XGBClassifier(**grid_search.best_params_, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    xgb_temp.fit(X_subset, y_subset)
    train_scores.append(xgb_temp.score(X_subset, y_subset))
    test_scores.append(xgb_temp.score(X_test, y_test))
ax3.plot([s*100 for s in train_sizes], train_scores, 'o-', label='Training Score', color='blue', linewidth=2)
ax3.plot([s*100 for s in train_sizes], test_scores, 'o-', label='Testing Score', color='red', linewidth=2)
ax3.set_xlabel('Training Set Size (%)')
ax3.set_ylabel('Accuracy')
ax3.set_title('Learning Curves - XGBoost')
ax3.legend()
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('02_results_part2.png', dpi=150, bbox_inches='tight')
plt.show()
# FINAL SUMMARY

print("\nFINAL SUMMARY")
print(f"""Model Performance:
- Decision Tree: Accuracy={results['Decision Tree']['accuracy']:.3f}, Precision={results['Decision Tree']['precision']:.3f}, Recall={results['Decision Tree']['recall']:.3f}, F1-Score={results['Decision Tree']['f1']:.3f}
- XGBoost: Accuracy={best_accuracy:.3f}, Precision={best_precision:.3f}, Recall={best_recall:.3f}, F1-Score={best_f1:.3f}""")

print("\nPROJECT COMPLETED SUCCESSFULLY")