"""
Fix remaining '34 batteries' references in source code cells across all notebooks.
Only fix markdown/comment text that says "34 batteries" as a count, not code that uses Battery_34 or '34' as a numeric value.
"""
import json, os, re, glob

PROJECT = r"d:\deeplearning\project ml"

def fix_source_text(text):
    """Replace count-of-batteries '34' references with '38' in markdown/comment text."""
    # Pattern: "34 batteries" in various forms (case insensitive)
    text = re.sub(r'\b34 batteries\b', '38 batteries (34 initial + 4 added)', text, flags=re.IGNORECASE)
    # "all 34 batteries"
    text = re.sub(r'\ball 34 batteries\b', 'all 38 batteries (34 initial + 4 added)', text, flags=re.IGNORECASE)
    # "34 / 34" (percent present)
    text = text.replace("34 / 34", "38 / 38")
    # "Batteries: 34"
    text = text.replace(": 34\n", ": 38\n")
    return text

notebooks_to_fix = {
    "01_mat_to_csv_conversion.ipynb": [11],  # markdown cells
    "02_dataset_understanding_and_eda.ipynb": [3, 19],
    "03_data_preprocessing.ipynb": [7],
    "04_soh_rul_calculation.ipynb": [14, 16],
    "05A_data/intermediate/final_dataset_analysis.ipynb": [2, 4, 9, 13, 20],
    "05B_target_validation_and_correction.ipynb": [2, 4, 7, 11, 16],
    "05_data/intermediate/final_dataset_construction.ipynb": [7, 10, 14],
}

for nb_file, cell_indices in notebooks_to_fix.items():
    nb_path = os.path.join(PROJECT, nb_file)
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)
    
    cells = nb["cells"]
    changed = False
    
    for ci in cell_indices:
        if ci >= len(cells):
            print(f"  ⚠ {nb_file} cell {ci} out of range (len={len(cells)})")
            continue
        
        cell = cells[ci]
        src = cell.get("source", [])
        
        if isinstance(src, list):
            full_src = "".join(src)
        else:
            full_src = src
        
        # Only fix markdown cells or comments in code
        if cell["cell_type"] == "markdown":
            new_src = fix_source_text(full_src)
            if new_src != full_src:
                cell["source"] = [new_src]
                print(f"  ✓ {nb_file} cell {ci} (markdown) fixed")
                changed = True
            else:
                print(f"  - {nb_file} cell {ci} (markdown) no change needed")
        elif cell["cell_type"] == "code":
            # For code cells, only fix comment lines (starting with #) or string literals in print()
            lines = full_src.split("\n")
            new_lines = []
            cell_changed = False
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("#") or "print(" in stripped:
                    new_line = fix_source_text(line)
                    if new_line != line:
                        cell_changed = True
                    new_lines.append(new_line)
                else:
                    new_lines.append(line)
            if cell_changed:
                cell["source"] = ["\n".join(new_lines)]
                print(f"  ✓ {nb_file} cell {ci} (code comments/prints) fixed")
                changed = True
            else:
                print(f"  - {nb_file} cell {ci} (code) no comment/print changes")
    
    if changed:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"  💾 Saved {nb_file}")
    
    print()

print("Done fixing source cells.")

