import re
from pathlib import Path

for fname in ["generate_dashboard.py", "calibrate_metrics.py", "update_status.py"]:
    p = Path(fname)
    if not p.exists(): continue
    content = p.read_text()

    # 1. Fix globbing
    content = content.replace('glob("B10_NPP*.yaml")', 'glob("B10_[RM]*.yaml")')
    
    # 2. Fix tag searching
    content = content.replace('startswith("B10_NPP")', 'startswith("B10_") and ("_N" in tag or "_M" in tag or "_R" in tag)')
    
    # 3. Fix regex matching and groups in generate_dashboard.py & calibrate_metrics.py
    if fname in ["generate_dashboard.py", "calibrate_metrics.py"]:
        content = re.sub(
            r'm = re\.search\(r"B10_NPP\(\\d\)\(\\d\)\(\\d\)", cfg_name\)\n\s+if not m:\n\s+continue\n\s+n_idx, p1_idx, p2_idx = m\.groups\(\)',
            r'm = re.search(r"B10_([RM])(\\d+)_N(\\d)P(\\d)(\\d)", cfg_name)\n    if not m:\n        continue\n    embed_type, dim, n_idx, p1_idx, p2_idx = m.groups()',
            content
        )
        content = re.sub(
            r'npp_match = re\.search\(r"B10_NPP\(\\d\)\(\\d\)\(\\d\)", cfg_name\)\n\s+if not npp_match:\n\s+continue\n\s+n_idx, p1_idx, p2_idx = npp_match\.groups\(\)',
            r'npp_match = re.search(r"B10_([RM])(\\d+)_N(\\d)P(\\d)(\\d)", cfg_name)\n    if not npp_match:\n        continue\n    embed_type, dim, n_idx, p1_idx, p2_idx = npp_match.groups()',
            content
        )
        
        # Add embed and dim to row dictionary in dashboard
        if fname == "generate_dashboard.py":
            content = content.replace(
                '"config": cfg_name,\n        "n_idx": n_idx,',
                '"config": cfg_name,\n        "embed_type": embed_type,\n        "dim": dim,\n        "n_idx": n_idx,'
            )

        # Fix noise maps (B10 has N1=10% and N2=30%)
        content = content.replace(
            '"1": "N1-Asym05",\n    "2": "N2-Asym15",\n    "3": "N3-Asym25",\n    "4": "N4-Ext10",\n    "5": "N5-Ext30",\n    "6": "N6-Ext50",',
            '"1": "N1-Ext10",\n    "2": "N2-Ext30",'
        )
        content = content.replace(
            '"1": "Asym 5% (asym05_ext0)",\n    "2": "Asym 15% (asym15_ext0)",\n    "3": "Asym 25% (asym25_ext0)",\n    "4": "Ext 10% (asym0_ext10)",\n    "5": "Ext 30% (asym0_ext30)",\n    "6": "Ext 50% (asym0_ext50)",',
            '"1": "Ext 10%",\n    "2": "Ext 30%",'
        )

    # 4. Fix project tag mismatch from sed
    content = content.replace('"B10_3DtoHD_fromB10"', '"B10_3DtoHD_fromB08"')

    p.write_text(content)

print("Patch applied.")
