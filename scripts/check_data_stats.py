import pandas as pd, os

df = pd.read_csv(r"d:\deeplearning\project ml\final_battery_dataset.csv")
print("=== final_battery_dataset.csv ===")
print(f"Shape: {df.shape}")
print(f"Batteries: {df['Battery_ID'].nunique()}")
print(f"Columns: {list(df.columns)}")
print()

ca = pd.read_csv(r"d:\deeplearning\project ml\battery_condition_analysis.csv")
print("=== battery_condition_analysis.csv ===")
print(f"Shape: {ca.shape}")
if "Battery_ID" in ca.columns:
    print(f"Batteries: {ca['Battery_ID'].nunique()}")
print(f"Columns: {list(ca.columns)}")
print(ca.head(5).to_string())
print()

# Check condition category counts
if "Condition_Category" in ca.columns:
    print("Condition categories:")
    print(ca["Condition_Category"].value_counts())
elif "condition_category" in ca.columns:
    print("Condition categories:")
    print(ca["condition_category"].value_counts())

