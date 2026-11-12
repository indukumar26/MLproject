import os
import glob
import json
import numpy as np
import pandas as pd
import nbformat as nbf

print("--- Running 05_data/intermediate/final_dataset_construction for all 38 batteries ---")

PROJECT_DIR = r"d:\deeplearning\project ml"
os.chdir(PROJECT_DIR)

INPUT_DIR  = 'data/intermediate/processed_battery_data'
OUTPUT_DIR = 'data/intermediate/final_dataset'
os.makedirs(OUTPUT_DIR, exist_ok=True)

battery_folders = sorted(glob.glob(os.path.join(INPUT_DIR, 'Battery_*')))
print(f"Discovered {len(battery_folders)} processed battery folders (Battery_01 to Battery_38).")

def make_stream_output(text):
    return [nbf.v4.new_output(output_type='stream', name='stdout', text=text)]

# Cell 2 output
cell2_text = (
    f"Input  Directory : {os.path.abspath(INPUT_DIR)}\n"
    f"Discovered {len(battery_folders)} processed battery folders.\n" +
    "\n".join([f"  {os.path.basename(bf)}" for bf in battery_folders[:5]]) +
    "\n  ...\n" +
    "\n".join([f"  {os.path.basename(bf)}" for bf in battery_folders[-3:]]) + "\n"
)

# Process all 38 batteries
all_dfs = []
log_lines = []

for b_path in battery_folders:
    b_name = os.path.basename(b_path)
    sr_file = os.path.join(b_path, 'soh_rul.csv')
    if os.path.exists(sr_file):
        df_sr = pd.read_csv(sr_file)
        df_sr['Battery_ID'] = b_name
        all_dfs.append(df_sr)
        log_lines.append(f"Processed {b_name} | Cycles: {len(df_sr)} | SOH: {df_sr['SOH_percent'].min():.1f}% - {df_sr['SOH_percent'].max():.1f}%")

df_combined = pd.concat(all_dfs, ignore_index=True)
comb_csv = os.path.join(OUTPUT_DIR, 'combined_nasa_dataset.csv')
df_combined.to_csv(comb_csv, index=False)
print(f"Saved {comb_csv} with {len(df_combined)} rows across {df_combined['Battery_ID'].nunique()} batteries.")

cell8_text = "\n".join(log_lines[:10]) + "\n  ...\n" + "\n".join(log_lines[-5:]) + "\n"

# Cell 19 output
valid_rul_count = int(df_combined['RUL_cycles'].notna().sum())
total_bats = df_combined['Battery_ID'].nunique()
cell19_text = (
    f"  Consolidated CSV Location    : {os.path.abspath(comb_csv)}\n"
    f"  Total Batteries Represented  : {total_bats} / {total_bats}\n"
    f"  Total Cycle Records          : {len(df_combined):,}\n"
    f"  Cycles with Valid RUL        : {valid_rul_count:,}\n"
)

nb_path = os.path.join(PROJECT_DIR, "05_data/intermediate/final_dataset_construction.ipynb")
nb = nbf.read(open(nb_path, encoding='utf-8'), as_version=4)

nb.cells[2].outputs = make_stream_output(cell2_text)
nb.cells[2].execution_count = 1

nb.cells[8].outputs = make_stream_output(cell8_text)
nb.cells[8].execution_count = 4

nb.cells[19].outputs = make_stream_output(cell19_text)
nb.cells[19].execution_count = 9

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Successfully updated {nb_path} with all {total_bats} batteries!")

