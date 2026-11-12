import pandas as pd
df = pd.read_csv(r"d:\deeplearning\project ml\data/final/corrected_data\corrected_final_battery_dataset.csv")
print("corrected_final_battery_dataset.csv:")
print(f"  Shape: {df.shape}")
print(f"  Batteries: {df['Battery_ID'].nunique()}")
print(f"  SOH>100: {(df['SOH_percent'] > 100).sum()}")
print(f"  SOH<=0: {(df['SOH_percent'] <= 0).sum()}")
print(f"  Valid RUL: {df['RUL_cycles'].notna().sum()}")
print(f"  Right-censored: {df['RUL_cycles'].isna().sum()}")
# EOL batteries
ca = pd.read_csv(r"d:\deeplearning\project ml\battery_condition_analysis.csv")
print(f"\n  Has_Valid_RUL (reached EOL): {ca['Has_Valid_RUL'].sum()}")
print(f"  Right-censored (no EOL): {(~ca['Has_Valid_RUL']).sum()}")

