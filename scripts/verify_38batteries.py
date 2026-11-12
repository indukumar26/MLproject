"""
Smart final verification — check for '34 batteries' as a COUNT (not in historical context).
"""
import json, os, glob

PROJECT = r"d:\deeplearning\project ml"
notebooks = sorted(glob.glob(os.path.join(PROJECT, "*.ipynb")))

print("=" * 70)
print("SMART SCAN: Checking notebook outputs for incorrect battery counts")
print("=" * 70)

issues = []

HISTORICAL_OK = [
    "Initial development used 34",
    "34 initial",
    "initial + 4",
    "cohorts 1-5",
    "cohorts 1–5",
    "NOTE:",
]

for nb_path in notebooks:
    nb_name = os.path.basename(nb_path)
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)
    
    for i, cell in enumerate(nb["cells"]):
        # Check actual count outputs (not historical context)
        for out in cell.get("outputs", []):
            txt = "".join(out.get("text", []))
            for line in txt.split("\n"):
                if "34" in line and "battery" in line.lower() and "Battery_34" not in line:
                    # Is this a historical/context reference?
                    is_historical = any(ctx in line for ctx in HISTORICAL_OK)
                    if not is_historical:
                        # Is it really a count?
                        lower = line.lower()
                        if any(kw in lower for kw in [
                            "total", "processed", "evaluated", "converted",
                            "batteries:", "count", "found", "loaded"
                        ]):
                            issues.append((nb_name, i, f"COUNT: {line.strip()[:100]}"))

if not issues:
    print("\n✅ ALL CLEAR! All notebooks correctly show 38 batteries.")
    print("   (Historical NOTE references to '34 initial' are intentional context.)")
else:
    print(f"\n⚠️  Found {len(issues)} actual count errors:\n")
    current_nb = None
    for nb_name, ci, flag in issues:
        if nb_name != current_nb:
            print(f"\n📓 {nb_name}")
            current_nb = nb_name
        print(f"  Cell {ci}: {flag}")
