import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV data
df = pd.read_csv('python/src/ann/motortraining/motor2_training_results.txt', delimiter='\t')

# Optional: Check column names
print("Column names:", df.columns.tolist())

# Extract relevant columns by index
trial_data = df.iloc[:, 1]           # 2nd column (Trial)
start_angle_data = df.iloc[:, 4]     # 5th column (Start Angle)
angle_change_data = df.iloc[:, 6]    # 7th column (Angle Change)

# Filter outliers using IQR method
Q1 = angle_change_data.quantile(0.25)
Q3 = angle_change_data.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.25 * IQR
upper_bound = Q3 + 1.25 * IQR

# Boolean mask for filtered data
filtered_mask = (angle_change_data >= lower_bound) & (angle_change_data <= upper_bound)

# Filtered data
filtered_trials = trial_data[filtered_mask]
filtered_start_angle = start_angle_data[filtered_mask]
filtered_angle_change = angle_change_data[filtered_mask]

# Set seaborn style
sns.set(style="whitegrid")

# Create subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 14))  # 3 rows, 1 column

# --- Plot 1: Line Plot ---
axes[0].plot(trial_data, angle_change_data, 'o-', label='Original', alpha=0.4)
axes[0].plot(filtered_trials, filtered_angle_change, 'o-', label='Filtered', color='blue')
axes[0].set_title('Angle Change vs Trial (With Outlier Filtering)')
axes[0].set_xlabel('Trial')
axes[0].set_ylabel('Angle Change')
axes[0].legend()
axes[0].grid(True)

# --- Plot 2: Boxplot ---
#sns.boxplot(y=angle_change_data, color='lightgray', width=0.5, ax=axes[1])
#sns.boxplot(y=filtered_angle_change, color='lightblue', width=0.3, ax=axes[1])
#axes[1].set_title('Boxplot of Angle Change (Gray = Original, Blue = Filtered)')
#axes[1].set_ylabel('Angle Change')
# --- Plot 2: Scatter Plot (Start Angle vs Angle Change) ---
axes[1].scatter(start_angle_data, angle_change_data, alpha=0.4, label='Original', color='gray')
axes[1].scatter(filtered_start_angle, filtered_angle_change, alpha=0.7, label='Filtered', color='blue')
axes[1].set_title('Angle Change vs Start Angle')
axes[1].set_xlabel('Start Angle')
axes[1].set_ylabel('Angle Change')
axes[1].legend()
axes[1].grid(True)

# --- Plot 3: Histogram + KDE ---
sns.histplot(angle_change_data, kde=True, color='gray', label='Original', bins=20,
             stat='density', alpha=0.4, ax=axes[2])
sns.histplot(angle_change_data, kde=True, color='blue', label='Filtered', bins=20,
             stat='density', alpha=0.6, ax=axes[2])
axes[2].set_title('Distribution of Angle Change (Original vs Filtered)')
axes[2].set_xlabel('Angle Change')
axes[2].set_ylabel('Density')
axes[2].legend()

# Layout
plt.tight_layout()
plt.show()


# Combine columns 1–6 (0-based index: 0 to 5) from original data for non-outliers
filtered_part = df.iloc[filtered_mask.values, 0:6].copy()

# Add 7th column: filtered angle change values
filtered_part['Angle Change'] = filtered_angle_change.values

# Add 8th column: constant value 12.5
filtered_part['Constant'] = 12.5

# Append to "results.txt" as tab-separated text
filtered_part.to_csv('python/src/ann/motortraining/results.txt', sep='\t', index=False, header=False, mode='a')

print("Filtered data with constant column appended to results.txt.")


