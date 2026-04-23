import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# STEP 0: Setup
sns.set_style("whitegrid")

# STEP 1: Load Dataset
print("---------- STEP 1: Loading Dataset ----------")
try:
    df = pd.read_csv("NP_dataset.csv")
    print(df.head())
except FileNotFoundError:
    print("Error: NP_dataset.csv not found in the current directory.")
    exit()

# STEP 2: Basic Understanding
print("\n---------- STEP 2: Basic Understanding ----------")
print("Shape:", df.shape)
print("\nInfo:")
df.info()
print("\nDescribe:")
print(df.describe())

# STEP 3: Missing Values
print("\n---------- STEP 3: Missing Values ----------")
print(df.isnull().sum())
df = df.dropna()

# STEP 4: Distribution Analysis
print("\n---------- STEP 4: Distribution Analysis ----------")
print("Close the plot window to continue...")
plt.figure(figsize=(15,10))
df.hist(bins=20)
plt.tight_layout()
plt.show()

# STEP 5: Target Variables Focus
print("\n---------- STEP 5: Target Variables Focus ----------")
print("Close the plot window to continue...")
# Check if columns exist before plotting
target_cols = ['particle_size', 'EE', 'LC']
if all(col in df.columns for col in target_cols):
    fig, ax = plt.subplots(1, 3, figsize=(18,5))

    sns.histplot(df['particle_size'], kde=True, ax=ax[0])
    ax[0].set_title("Particle Size")

    sns.histplot(df['EE'], kde=True, ax=ax[1])
    ax[1].set_title("Encapsulation Efficiency")

    sns.histplot(df['LC'], kde=True, ax=ax[2])
    ax[2].set_title("Loading Capacity")

    plt.show()
else:
    print("Warning: Target columns 'particle_size', 'EE', or 'LC' not found for Step 5.")

# STEP 6: Correlation Heatmap
print("\n---------- STEP 6: Correlation Heatmap ----------")
print("Close the plot window to continue...")
plt.figure(figsize=(12,10))
# select only numeric columns to avoid warnings in newer pandas versions
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()

sns.heatmap(corr, cmap='coolwarm', annot=False)
plt.title("Correlation Matrix")
plt.show()

# STEP 7: Top Correlations with Target
print("\n---------- STEP 7: Top Correlations with Target ----------")
if 'particle_size' in corr.columns:
    corr_target = corr['particle_size'].sort_values(ascending=False)
    print("Top correlations for 'particle_size':")
    print(corr_target)

if 'EE' in corr.columns:
    print("\nTop correlations for 'EE':")
    print(corr['EE'].sort_values(ascending=False))

if 'LC' in corr.columns:
    print("\nTop correlations for 'LC':")
    print(corr['LC'].sort_values(ascending=False))

# STEP 8: Scatter Plots
print("\n---------- STEP 8: Scatter Plots ----------")
print("Close the plot windows to continue...")
if 'drug/polymer' in df.columns and 'LC' in df.columns:
    sns.scatterplot(x='drug/polymer', y='LC', data=df)
    plt.title("Drug/Polymer vs Loading Capacity")
    plt.show()

if 'surfactant_concentration' in df.columns and 'particle_size' in df.columns:
    sns.scatterplot(x='surfactant_concentration', y='particle_size', data=df)
    plt.title("Surfactant Concentration vs Particle Size")
    plt.show()

if 'solvent_polarity_index' in df.columns and 'EE' in df.columns:
    sns.scatterplot(x='solvent_polarity_index', y='EE', data=df)
    plt.title("Solvent Polarity Index vs Encapsulation Efficiency")
    plt.show()

# STEP 9: Pairplot
print("\n---------- STEP 9: Pairplot ----------")
print("Close the plot window to continue...")
pairplot_cols = ['particle_size','EE','LC','drug/polymer','surfactant_concentration']
valid_cols = [col for col in pairplot_cols if col in df.columns]
if len(valid_cols) > 1:
    sns.pairplot(df[valid_cols])
    plt.title("Pairplot of selected variables")
    plt.show()
else:
    print("Warning: Columns missing for Pairplot.")


# STEP 10: Feature Importance (ML Insight)
print("\n---------- STEP 10: Feature Importance ----------")
print("Close the plot window to continue...")
# check if relevant columns exist
if all(col in df.columns for col in ['particle_size', 'EE', 'LC']):
    X = numeric_df.drop(['particle_size','EE','LC'], axis=1, errors='ignore')
    y = numeric_df['particle_size']

    # Make sure X is not empty
    if not X.empty:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        importance = pd.Series(model.feature_importances_, index=X.columns)
        importance.sort_values().plot(kind='barh', figsize=(10,6))
        plt.title("Feature Importance for Particle Size")
        plt.show()

        # STEP 11: Model Performance
        print("\n---------- STEP 11: Model Performance ----------")
        y_pred = model.predict(X_test)
        print("R2 Score for Particle Size prediction:", r2_score(y_test, y_pred))
else:
    print("Warning: Target variables missing for Random Forest Analysis.")

print("\n✅ Script execution completed.")
