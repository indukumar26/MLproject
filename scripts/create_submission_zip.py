import os
import zipfile

zip_filename = "Battery_Prognostics_Submission.zip"

include_dirs = [
    ('notebooks', 'notebooks'),
    ('models', 'models'),
    ('data/final', 'data/final'),
    ('data/intermediate/calce_prepared_data', 'data/intermediate/calce_prepared_data'),
    ('report_assets', 'report_assets'),
    ('scripts', 'scripts'),
]

include_files = [
    'FINAL_PROJECT_REPORT.md',
    'PROJECT_PIPELINE.md',
    'README.md',
    'requirements.txt',
    'run_live_test_evaluation.py'
]

print(f"Creating submission archive: {zip_filename} ...")
with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for src_dir, arc_dir in include_dirs:
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, src_dir)
                archive_name = os.path.join(arc_dir, rel_path)
                zipf.write(full_path, archive_name)
    
    for f in include_files:
        if os.path.exists(f):
            zipf.write(f, f)

size_mb = os.path.getsize(zip_filename) / (1024 * 1024)
print(f"Submission archive created successfully: {zip_filename} ({size_mb:.2f} MB)")
