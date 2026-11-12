"""
rebuild_splits_b24_to_train.py
Constructs the quality-filtered NASA battery dataset with Battery_24 moved to Train.
Train (22), Validation (6), Test (4): Battery_04, Battery_16, Battery_30, Battery_38.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

BASE_DIR = r"d:\deeplearning\project ml"
OUTPUT_DIR = os.path.join(BASE_DIR, "data/final/filtered_lstm_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Battery Cohort Definition
EXCLUDED_BATTERIES = {
    'Battery_17': 'Partial reference capacity at cycle 1 (ref=1.161 Ah, max=1.885 Ah, SOH reaches 162.33%)',
    'Battery_19': 'Conditioning-cycle reference error and severe capacity spikes (ref=1.002 Ah, spike to 2.444 Ah at cycle 279, SOH reaches 243.92%)',
    'Battery_20': 'Partial reference capacity at cycle 1 (ref=1.103 Ah, max=1.786 Ah, SOH reaches 161.94%)',
    'Battery_31': 'Abnormal capacity measurements exceeding physical cell rating of 2.0 Ah (ref=1.421 Ah, max=2.379 Ah, SOH reaches 167.45%)',
    'Battery_32': 'Erratic capacity glitches and impossible measurements (max=2.640 Ah, extreme jumps, SOH reaches 169.57%)',
    'Battery_33': 'Abnormal capacity measurements exceeding physical cell rating of 2.0 Ah (ref=1.293 Ah, max=2.330 Ah, SOH reaches 180.20%)'
}

RETAINED_BATTERIES = [
    'Battery_01', 'Battery_02', 'Battery_03', 'Battery_04', 'Battery_05',
    'Battery_06', 'Battery_07', 'Battery_08', 'Battery_09', 'Battery_10',
    'Battery_11', 'Battery_12', 'Battery_13', 'Battery_14', 'Battery_15',
    'Battery_16', 'Battery_18', 'Battery_21', 'Battery_22', 'Battery_23',
    'Battery_24', 'Battery_25', 'Battery_26', 'Battery_27', 'Battery_28',
    'Battery_29', 'Battery_30', 'Battery_34', 'Battery_35', 'Battery_36',
    'Battery_37', 'Battery_38'
]

DISCHARGE_FEATURES = [
    'ambient_temperature',
    'discharge_duration_s',
    'voltage_measured_mean',
    'voltage_measured_min',
    'voltage_measured_max',
    'voltage_measured_std',
    'current_measured_mean',
    'current_measured_min',
    'temperature_measured_mean',
    'temperature_measured_max',
    'temperature_measured_min',
    'voltage_load_mean',
    'current_load_mean'
]

SEQUENCE_LENGTH = 5

# Split assignments with Battery_24 moved to Train
SPLIT_ASSIGNMENTS = {
    # Train (22 batteries) - Battery_24 moved here
    'Battery_01': 'Train',
    'Battery_02': 'Train',
    'Battery_03': 'Train',
    'Battery_05': 'Train',
    'Battery_07': 'Train',
    'Battery_08': 'Train',
    'Battery_09': 'Train',
    'Battery_10': 'Train',
    'Battery_11': 'Train',
    'Battery_12': 'Train',
    'Battery_13': 'Train',
    'Battery_14': 'Train',
    'Battery_18': 'Train',
    'Battery_22': 'Train',
    'Battery_23': 'Train', # Provisional flagged
    'Battery_24': 'Train', # MOVED FROM TEST TO TRAIN
    'Battery_26': 'Train',
    'Battery_27': 'Train',
    'Battery_28': 'Train',
    'Battery_34': 'Train',
    'Battery_35': 'Train',
    'Battery_36': 'Train',
    # Validation (6 batteries)
    'Battery_06': 'Validation',
    'Battery_15': 'Validation',
    'Battery_21': 'Validation',
    'Battery_25': 'Validation',
    'Battery_29': 'Validation',
    'Battery_37': 'Validation',
    # Test (4 batteries)
    'Battery_04': 'Test',
    'Battery_16': 'Test',
    'Battery_30': 'Test',
    'Battery_38': 'Test'
}

train_bats = [b for b, s in SPLIT_ASSIGNMENTS.items() if s == 'Train']
val_bats   = [b for b, s in SPLIT_ASSIGNMENTS.items() if s == 'Validation']
test_bats  = [b for b, s in SPLIT_ASSIGNMENTS.items() if s == 'Test']

print(f"Total Retained: {len(RETAINED_BATTERIES)}")
print(f"Train: {len(train_bats)} batteries: {train_bats}")
print(f"Validation: {len(val_bats)} batteries: {val_bats}")
print(f"Test: {len(test_bats)} batteries: {test_bats}")

assert len(RETAINED_BATTERIES) == 32
assert len(train_bats) == 22
assert len(val_bats) == 6
assert len(test_bats) == 4

# 2. Load Source Files
df_corr_source = pd.read_csv(os.path.join(BASE_DIR, "data/final/corrected_data", "corrected_final_battery_dataset.csv"))
df_soh_source  = pd.read_csv(os.path.join(BASE_DIR, "battery_cycle_capacity_soh.csv"))
df_ref_source  = pd.read_csv(os.path.join(BASE_DIR, "battery_reference_capacity_corrected.csv"))

# 3. Filter to Retained Batteries
df_ref_filtered = df_ref_source[df_ref_source['Battery_ID'].isin(RETAINED_BATTERIES)].copy().sort_values('Battery_ID').reset_index(drop=True)
df_soh_filtered = df_soh_source[df_soh_source['Battery_ID'].isin(RETAINED_BATTERIES)].copy().sort_values(['Battery_ID', 'Cycle_Number']).reset_index(drop=True)
df_features_filtered = df_corr_source[df_corr_source['Battery_ID'].isin(RETAINED_BATTERIES)].copy().sort_values(['Battery_ID', 'Cycle']).reset_index(drop=True)

# Merge verified SOH and Capacity into feature dataset
merged_df = pd.merge(
    df_features_filtered[['Battery_ID', 'Cycle', 'cycle_index'] + DISCHARGE_FEATURES],
    df_soh_filtered[['Battery_ID', 'cycle_index', 'capacity_Ahr', 'C_ref_Ahr', 'SOH_percent', 'Is_Glitch']],
    on=['Battery_ID', 'cycle_index'],
    how='left'
)

merged_df['Cycle_Number'] = merged_df['Cycle']

# 4. Compute EOL and RUL
eol_map = {}
for b_id, grp in merged_df.groupby('Battery_ID'):
    grp_sorted = grp.sort_values('Cycle_Number')
    sub_eol = grp_sorted[grp_sorted['SOH_percent'] <= 80.0]
    if len(sub_eol) > 0:
        eol_cycle = int(sub_eol['Cycle_Number'].iloc[0])
        status = 'Observed'
    else:
        eol_cycle = np.nan
        status = 'Right-Censored'
    eol_map[b_id] = {'EOL_Cycle': eol_cycle, 'EOL_Status': status}

merged_df['EOL_Cycle'] = merged_df['Battery_ID'].map(lambda b: eol_map[b]['EOL_Cycle'])
merged_df['EOL_Status'] = merged_df['Battery_ID'].map(lambda b: eol_map[b]['EOL_Status'])

def calc_rul(row):
    if pd.isna(row['EOL_Cycle']):
        return np.nan
    return max(float(row['EOL_Cycle']) - float(row['Cycle_Number']), 0.0)

merged_df['RUL_cycles'] = merged_df.apply(calc_rul, axis=1)
merged_df.loc[merged_df['Is_Glitch'] == True, 'SOH_percent'] = np.nan

final_cols = [
    'Battery_ID', 'Cycle_Number', 'cycle_index',
    'capacity_Ahr', 'C_ref_Ahr', 'SOH_percent',
    'EOL_Cycle', 'EOL_Status', 'RUL_cycles', 'Is_Glitch'
] + DISCHARGE_FEATURES

filtered_dataset = merged_df[final_cols].copy()

# Save datasets to OUTPUT_DIR
dataset_csv_path = os.path.join(OUTPUT_DIR, "filtered_nasa_battery_dataset.csv")
filtered_dataset.to_csv(dataset_csv_path, index=False)
print(f"[Saved] {dataset_csv_path}")

soh_csv_path = os.path.join(OUTPUT_DIR, "filtered_battery_cycle_capacity_soh.csv")
df_soh_filtered.to_csv(soh_csv_path, index=False)

ref_csv_path = os.path.join(OUTPUT_DIR, "filtered_battery_reference_capacity.csv")
df_ref_filtered.to_csv(ref_csv_path, index=False)

# Quality summary
quality_records = []
for b_id in RETAINED_BATTERIES:
    sub = filtered_dataset[filtered_dataset['Battery_ID'] == b_id]
    ref_val = sub['C_ref_Ahr'].iloc[0]
    total_c = len(sub)
    glitches = sub['Is_Glitch'].sum()
    valid_soh = sub['SOH_percent'].dropna()
    min_soh = valid_soh.min() if len(valid_soh) > 0 else np.nan
    max_soh = valid_soh.max() if len(valid_soh) > 0 else np.nan
    mean_soh = valid_soh.mean() if len(valid_soh) > 0 else np.nan
    soh_gt_100 = (valid_soh > 100.0).sum()
    soh_gt_110 = (valid_soh > 110.0).sum()
    eol_c = sub['EOL_Cycle'].iloc[0]
    eol_stat = sub['EOL_Status'].iloc[0]
    
    notes = []
    if b_id == 'Battery_23':
        notes.append('Provisional: Reference capacity at cycle 43 (cycle_index 104); early cycles SOH~4%')
    if b_id == 'Battery_24':
        notes.append('Moved to Train: cold plunge operating condition represented in training')
    if max_soh > 110.0:
        notes.append(f'SOH anomaly >110% (max {max_soh:.2f}%)')
    elif max_soh > 100.0:
        notes.append(f'SOH slightly exceeds 100% (max {max_soh:.2f}%)')
    if glitches > 0:
        notes.append(f'{glitches} glitch cycle(s) set to NaN')
    if total_c < SEQUENCE_LENGTH:
        notes.append(f'Insufficient cycles ({total_c} < {SEQUENCE_LENGTH}) for LSTM window')
        
    quality_records.append({
        'Battery_ID': b_id,
        'Split': SPLIT_ASSIGNMENTS[b_id],
        'Total_Cycles': total_c,
        'C_ref_Ahr': ref_val,
        'Glitch_Cycles': glitches,
        'Valid_SOH_Cycles': len(valid_soh),
        'Min_SOH_%': min_soh,
        'Max_SOH_%': max_soh,
        'Mean_SOH_%': mean_soh,
        'Cycles_SOH_gt_100': soh_gt_100,
        'Cycles_SOH_gt_110': soh_gt_110,
        'EOL_Cycle': eol_c,
        'EOL_Status': eol_stat,
        'Quality_Notes': '; '.join(notes) if notes else 'Clean'
    })

df_quality = pd.DataFrame(quality_records)
quality_csv_path = os.path.join(OUTPUT_DIR, "battery_quality_summary.csv")
df_quality.to_csv(quality_csv_path, index=False)
print(f"[Saved] {quality_csv_path}")

# Split assignments
split_records = []
for b_id in RETAINED_BATTERIES:
    total_c = len(filtered_dataset[filtered_dataset['Battery_ID'] == b_id])
    split_records.append({
        'Battery_ID': b_id,
        'Split': SPLIT_ASSIGNMENTS[b_id],
        'Total_Cycles': total_c
    })
df_split = pd.DataFrame(split_records)
split_csv_path = os.path.join(OUTPUT_DIR, "split_battery_assignments.csv")
df_split.to_csv(split_csv_path, index=False)
print(f"[Saved] {split_csv_path}")

# 5. Generate Sliding Window Sequences (length = 5)
def generate_sequences(df_sub, split_name):
    X_list = []
    y_soh_list = []
    y_rul_list = []
    mask_soh_list = []
    mask_rul_list = []
    metadata_list = []
    
    seq_counter = 0
    
    for b_id, grp in df_sub.groupby('Battery_ID'):
        grp_sorted = grp.sort_values('Cycle_Number').reset_index(drop=True)
        n_cycles = len(grp_sorted)
        if n_cycles < SEQUENCE_LENGTH:
            print(f"  [Notice] {b_id} in {split_name} has only {n_cycles} cycles (< {SEQUENCE_LENGTH}). Skipping sequence generation.")
            continue
            
        features_mat = grp_sorted[DISCHARGE_FEATURES].values
        soh_targets = grp_sorted['SOH_percent'].values
        rul_targets = grp_sorted['RUL_cycles'].values
        cycle_nums = grp_sorted['Cycle_Number'].values
        
        for i in range(n_cycles - SEQUENCE_LENGTH + 1):
            window_x = features_mat[i : i + SEQUENCE_LENGTH]
            final_idx = i + SEQUENCE_LENGTH - 1
            
            soh_targ = soh_targets[final_idx]
            rul_targ = rul_targets[final_idx]
            start_c = cycle_nums[i]
            end_c = cycle_nums[final_idx]
            
            # SOH mask
            if pd.notna(soh_targ):
                y_soh_val = float(soh_targ)
                m_soh_val = 1.0
            else:
                y_soh_val = -1.0
                m_soh_val = 0.0
                
            # RUL mask
            if pd.notna(rul_targ):
                y_rul_val = float(rul_targ)
                m_rul_val = 1.0
            else:
                y_rul_val = -1.0
                m_rul_val = 0.0
                
            X_list.append(window_x)
            y_soh_list.append(y_soh_val)
            y_rul_list.append(y_rul_val)
            mask_soh_list.append(m_soh_val)
            mask_rul_list.append(m_rul_val)
            
            metadata_list.append({
                'seq_id': seq_counter,
                'Split': split_name,
                'Battery_ID': b_id,
                'Start_Cycle': int(start_c),
                'End_Cycle': int(end_c),
                'SOH_Target': y_soh_val if m_soh_val == 1.0 else np.nan,
                'RUL_Target': y_rul_val if m_rul_val == 1.0 else np.nan,
                'Has_SOH': bool(m_soh_val == 1.0),
                'Has_RUL': bool(m_rul_val == 1.0)
            })
            seq_counter += 1
            
    X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32)
    y_soh_tensor = torch.tensor(np.array(y_soh_list), dtype=torch.float32).unsqueeze(-1)
    y_rul_tensor = torch.tensor(np.array(y_rul_list), dtype=torch.float32).unsqueeze(-1)
    mask_soh_tensor = torch.tensor(np.array(mask_soh_list), dtype=torch.float32).unsqueeze(-1)
    mask_rul_tensor = torch.tensor(np.array(mask_rul_list), dtype=torch.float32).unsqueeze(-1)
    meta_df = pd.DataFrame(metadata_list)
    
    return {
        'X': X_tensor,
        'y_soh': y_soh_tensor,
        'y_rul': y_rul_tensor,
        'mask_soh': mask_soh_tensor,
        'mask_rul': mask_rul_tensor,
        'metadata': meta_df
    }

train_df = filtered_dataset[filtered_dataset['Battery_ID'].map(SPLIT_ASSIGNMENTS) == 'Train']
val_df   = filtered_dataset[filtered_dataset['Battery_ID'].map(SPLIT_ASSIGNMENTS) == 'Validation']
test_df  = filtered_dataset[filtered_dataset['Battery_ID'].map(SPLIT_ASSIGNMENTS) == 'Test']

print("Generating sequences...")
train_pkg = generate_sequences(train_df, 'Train')
val_pkg   = generate_sequences(val_df, 'Validation')
test_pkg  = generate_sequences(test_df, 'Test')

print(f"Train sequences: {train_pkg['X'].shape}")
print(f"Val sequences:   {val_pkg['X'].shape}")
print(f"Test sequences:  {test_pkg['X'].shape}")

# Save Metadata CSVs
train_meta_path = os.path.join(OUTPUT_DIR, "train_sequences_metadata.csv")
val_meta_path   = os.path.join(OUTPUT_DIR, "val_sequences_metadata.csv")
test_meta_path  = os.path.join(OUTPUT_DIR, "test_sequences_metadata.csv")

train_pkg['metadata'].to_csv(train_meta_path, index=False)
val_pkg['metadata'].to_csv(val_meta_path, index=False)
test_pkg['metadata'].to_csv(test_meta_path, index=False)
print(f"[Saved] Metadata CSVs to {OUTPUT_DIR}")

# Save PyTorch Tensors
torch.save({
    'X': train_pkg['X'],
    'y_soh': train_pkg['y_soh'],
    'y_rul': train_pkg['y_rul'],
    'mask_soh': train_pkg['mask_soh'],
    'mask_rul': train_pkg['mask_rul']
}, os.path.join(OUTPUT_DIR, "train_tensors.pt"))

torch.save({
    'X': val_pkg['X'],
    'y_soh': val_pkg['y_soh'],
    'y_rul': val_pkg['y_rul'],
    'mask_soh': val_pkg['mask_soh'],
    'mask_rul': val_pkg['mask_rul']
}, os.path.join(OUTPUT_DIR, "val_tensors.pt"))

torch.save({
    'X': test_pkg['X'],
    'y_soh': test_pkg['y_soh'],
    'y_rul': test_pkg['y_rul'],
    'mask_soh': test_pkg['mask_soh'],
    'mask_rul': test_pkg['mask_rul']
}, os.path.join(OUTPUT_DIR, "test_tensors.pt"))
print(f"[Saved] train_tensors.pt, val_tensors.pt, test_tensors.pt")

# 6. Fit StandardScaler Strictly on Training Sequences (Now includes Battery_24!)
X_train_np = train_pkg['X'].numpy()
N_train, W, D = X_train_np.shape
X_train_2d = X_train_np.reshape(-1, D)

scaler = StandardScaler()
scaler.fit(X_train_2d)

scaler_path = os.path.join(OUTPUT_DIR, "feature_scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"[Saved] {scaler_path} (Fitted on {N_train} sequences across 22 train batteries)")

# Save preparation_config.json
train_bats = sorted(df_split[df_split['Split'] == 'Train']['Battery_ID'].tolist())
val_bats   = sorted(df_split[df_split['Split'] == 'Validation']['Battery_ID'].tolist())
test_bats  = sorted(df_split[df_split['Split'] == 'Test']['Battery_ID'].tolist())

prep_config = {
    'dataset_version': 'Quality-Filtered NASA PCoE Dataset (32 Batteries) - Battery_24 in Train',
    'split_strategy': 'Condition-Stratified Battery-Wise (Battery_24 in Train, 4 Test Batteries)',
    'sequence_length': SEQUENCE_LENGTH,
    'number_of_features': len(DISCHARGE_FEATURES),
    'feature_names': DISCHARGE_FEATURES,
    'target_names': ['SOH_percent', 'RUL_cycles'],
    'random_seed': 42,
    'total_batteries_retained': len(RETAINED_BATTERIES),
    'total_batteries_excluded': len(EXCLUDED_BATTERIES),
    'excluded_batteries': sorted(list(EXCLUDED_BATTERIES.keys())),
    'train_batteries': train_bats,
    'validation_batteries': val_bats,
    'test_batteries': test_bats,
    'train_battery_count': len(train_bats),
    'validation_battery_count': len(val_bats),
    'test_battery_count': len(test_bats),
    'train_cycles': int(len(train_df)),
    'val_cycles': int(len(val_df)),
    'test_cycles': int(len(test_df)),
    'train_sequences_shape': list(train_pkg['X'].shape),
    'val_sequences_shape': list(val_pkg['X'].shape),
    'test_sequences_shape': list(test_pkg['X'].shape),
    'feature_scaler_path': 'data/final/filtered_lstm_data/feature_scaler.pkl',
    'provisional_batteries': ['Battery_23'],
    'provisional_battery_notes': {
        'Battery_23': 'Reference capacity selected at cycle 43 (cycle_index 104); early cycles exhibit SOH ~4%.'
    },
    'cold_plunge_stratification': {
        'Train_representatives': ['Battery_23', 'Battery_24', 'Battery_26'],
        'Validation_representative': 'Battery_25',
        'Test_target': 'None (Battery_24 moved to Train)'
    }
}

config_json_path = os.path.join(OUTPUT_DIR, "preparation_config.json")
with open(config_json_path, 'w') as f:
    json.dump(prep_config, f, indent=2)
print(f"[Saved] {config_json_path}")

print("\n" + "="*70)
print("SPLIT REBUILD COMPLETE: Battery_24 in Train, 4 Test Batteries")
print("="*70)
print(f"Train : {len(train_bats)} batteries -> {train_pkg['X'].shape[0]} sequences")
print(f"Val   : {len(val_bats)} batteries -> {val_pkg['X'].shape[0]} sequences")
print(f"Test  : {len(test_bats)} batteries -> {test_pkg['X'].shape[0]} sequences")
print(f"Test Batteries: {test_bats}")
