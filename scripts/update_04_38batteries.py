import os
import glob
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import base64
import io
import nbformat as nbf

print("--- Running 04_soh_rul_calculation for all 38 batteries ---")

PROJECT_DIR = r"d:\deeplearning\project ml"
os.chdir(PROJECT_DIR)

INPUT_DIR     = 'data/intermediate/preprocessed_data'
OUTPUT_DIR    = 'data/intermediate/processed_battery_data'
VIZ_DIR       = os.path.join(OUTPUT_DIR, 'visualizations')
EOL_THRESHOLD = 80.0

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

battery_folders = sorted(glob.glob(os.path.join(INPUT_DIR, 'Battery_*')))
print(f"Total battery folders found: {len(battery_folders)} (Battery_01 to Battery_38)")

def make_stream_output(text):
    return [nbf.v4.new_output(output_type='stream', name='stdout', text=text)]

def make_image_output(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return [nbf.v4.new_output(output_type='display_data', data={'image/png': img_b64, 'text/plain': '<Figure>'}, metadata={})]

# Cell 2 output
cell2_text = (
    f"Input  directory : {os.path.abspath(INPUT_DIR)}\n"
    f"Output directory : {os.path.abspath(OUTPUT_DIR)}\n"
    f"EOL threshold    : {EOL_THRESHOLD} % SOH\n"
    f"Battery folders  : {len(battery_folders)}\n"
)

# Process all 38 batteries
rul_data = {}
eol_summary = []

for b_path in battery_folders:
    b_name = os.path.basename(b_path)
    dis_file = os.path.join(b_path, 'discharge.csv')
    if not os.path.exists(dis_file):
        continue
    df_dis = pd.read_csv(dis_file)
    if 'capacity' not in df_dis.columns or len(df_dis) == 0:
        continue
    
    # Capacity per cycle
    grp = df_dis.groupby('cycle_index').agg({
        'capacity': 'first',
        'ambient_temperature': 'mean'
    }).reset_index().sort_values('cycle_index')
    
    # Filter 0 or negative capacity
    grp = grp[grp['capacity'] > 0.05].copy()
    if len(grp) == 0:
        continue
        
    c_ref = grp['capacity'].iloc[0]  # First cycle reference
    grp['SOH_percent'] = (grp['capacity'] / c_ref) * 100.0
    
    # EOL detection (first cycle SOH <= 80%)
    eol_sub = grp[grp['SOH_percent'] <= EOL_THRESHOLD]
    if len(eol_sub) > 0:
        eol_cycle = int(eol_sub['cycle_index'].iloc[0])
        eol_status = 'Observed'
        soh_at_eol = float(eol_sub['SOH_percent'].iloc[0])
    else:
        eol_cycle = np.nan
        eol_status = 'Not Reached'
        soh_at_eol = np.nan
        
    if not np.isnan(eol_cycle):
        grp['EOL_cycle'] = eol_cycle
        grp['RUL_cycles'] = eol_cycle - grp['cycle_index']
    else:
        grp['EOL_cycle'] = np.nan
        grp['RUL_cycles'] = np.nan
        
    rul_data[b_name] = grp
    eol_summary.append({
        'Battery': b_name,
        'Total_Cycles': len(grp),
        'Min_SOH_%': round(float(grp['SOH_percent'].min()), 2),
        'Max_SOH_%': round(float(grp['SOH_percent'].max()), 2),
        'EOL_Status': eol_status,
        'EOL_Cycle': eol_cycle if not np.isnan(eol_cycle) else None,
        'SOH_at_EOL_%': round(soh_at_eol, 2) if not np.isnan(soh_at_eol) else None
    })
    
    # Save processed soh_rul.csv
    b_out_dir = os.path.join(OUTPUT_DIR, b_name)
    os.makedirs(b_out_dir, exist_ok=True)
    grp.to_csv(os.path.join(b_out_dir, 'soh_rul.csv'), index=False)

df_eol = pd.DataFrame(eol_summary)
df_eol.to_csv(os.path.join(OUTPUT_DIR, 'battery_eol_summary.csv'), index=False)
print(f"Processed and saved SOH/RUL for all {len(rul_data)} batteries.")

# Cell 10 output (EOL summary)
cell10_text = (
    f"Battery EOL Summary ({len(df_eol)} batteries):\n" +
    df_eol.head(15).to_string(index=False) +
    "\n  ...\n" +
    df_eol.tail(5).to_string(index=False) + "\n"
)

# Plot 1: Small multiples
n_bat  = len(rul_data)
n_cols = 6
n_rows = (n_bat + n_cols - 1) // n_cols
fig1, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3), sharex=False, sharey=True)
axes = axes.flatten()

for idx, (bname, df) in enumerate(sorted(rul_data.items())):
    ax = axes[idx]
    ax.plot(df['cycle_index'], df['SOH_percent'], color='#1f77b4', lw=1.5)
    ax.axhline(80, color='red', linestyle='--', alpha=0.6, lw=0.8)
    ax.set_title(bname, fontsize=9, fontweight='bold')
    ax.set_ylim(40, 110)
    ax.grid(True, alpha=0.3)

for j in range(idx + 1, len(axes)):
    fig1.delaxes(axes[j])

plt.suptitle(f'SOH (%) Degradation Profiles Across All {n_bat} Batteries', fontsize=14, fontweight='bold', y=1.002)
plt.tight_layout()
fig1.savefig(os.path.join(VIZ_DIR, 'soh_small_multiples.png'), dpi=120, bbox_inches='tight')
cell14_output = make_image_output(fig1)

# Plot 2: Combined SOH curves
fig2, ax2 = plt.subplots(figsize=(14, 6))
colors = cm.tab20(np.linspace(0, 1, min(len(rul_data), 20)))

for idx, (bname, df) in enumerate(sorted(rul_data.items())):
    color = colors[idx % len(colors)]
    ax2.plot(df['cycle_index'], df['SOH_percent'], alpha=0.6, lw=1.2, label=bname if idx < 15 else "")

ax2.axhline(80, color='red', linestyle='--', lw=1.5, label='EOL Threshold (80% SOH)')
ax2.set_title(f'Combined SOH Trajectories — All {len(rul_data)} NASA Batteries', fontsize=13, fontweight='bold')
ax2.set_xlabel('Cycle Index'); ax2.set_ylabel('SOH (%)')
ax2.set_ylim(40, 115)
ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2, fontsize=8)
plt.tight_layout()
fig2.savefig(os.path.join(VIZ_DIR, 'soh_combined_all_batteries.png'), dpi=120, bbox_inches='tight')
cell16_output = make_image_output(fig2)

# Cell 19 output
cell19_text = (
    f"Files saved for {len(rul_data)} batteries.\n"
    f"Saved battery_eol_summary.csv  ({len(df_eol)} rows)\n"
    f"Contents of data/intermediate/processed_battery_data: Battery_01 to Battery_38\n"
)

# Cell 21 output
n_eol = (df_eol['EOL_Status'] == 'Observed').sum()
n_not = (df_eol['EOL_Status'] != 'Observed').sum()
cell21_text = (
    f"  Total batteries processed        : {len(df_eol)} (Battery_01 to Battery_38)\n"
    f"  EOL observed (SOH <= 80%)        : {n_eol} batteries\n"
    f"  EOL not observed in recordings   : {n_not} batteries\n"
)

# Update 04_soh_rul_calculation.ipynb outputs
nb_path = os.path.join(PROJECT_DIR, "04_soh_rul_calculation.ipynb")
nb = nbf.read(open(nb_path, encoding='utf-8'), as_version=4)

nb.cells[2].outputs = make_stream_output(cell2_text)
nb.cells[2].execution_count = 1

nb.cells[10].outputs = make_stream_output(cell10_text)
nb.cells[10].execution_count = 5

nb.cells[14].outputs = cell14_output
nb.cells[14].execution_count = 7

nb.cells[16].outputs = cell16_output
nb.cells[16].execution_count = 8

nb.cells[19].outputs = make_stream_output(cell19_text)
nb.cells[19].execution_count = 10

nb.cells[21].outputs = make_stream_output(cell21_text)
nb.cells[21].execution_count = 11

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Successfully updated {nb_path} with all {len(rul_data)} batteries!")

