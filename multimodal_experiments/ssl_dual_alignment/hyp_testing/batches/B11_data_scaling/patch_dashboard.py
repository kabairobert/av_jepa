import re
from pathlib import Path

b10_path = Path("/home/rkabai/github/eb_jepa_private/multimodal_experiments/ssl_dual_alignment/hyp_testing/batches/B10_3DtoHD_fromB08/generate_dashboard.py")
b11_path = Path("/home/rkabai/github/eb_jepa_private/multimodal_experiments/ssl_dual_alignment/hyp_testing/batches/B11_data_scaling/generate_dashboard.py")

content = b10_path.read_text()

# 1. Update paths and globals
content = content.replace('batch_id = "B10_3DtoHD_fromB08"', 'batch_id = "B11_data_scaling"')
content = content.replace('glob("B10_[RM]*.yaml")', 'glob("B11_*.yaml")')
content = content.replace(
    'cfg_tag = next((tag for tag in r.tags if tag.startswith("B10_") and ("_N" in tag or "_M" in tag or "_R" in tag)), None)',
    'cfg_tag = next((tag for tag in r.tags if tag.startswith("B11_")), None)'
)
content = content.replace('B10 Volumetric Sweep Dashboard', 'B11 Data Scaling Dashboard')
content = content.replace('<h1><span>B10</span> Volumetric Sweep Dashboard</h1>', '<h1><span>B11</span> Data Scaling Dashboard</h1>')

# 2. Update config parser regex
old_regex = r'npp_match = re\.search\(r"B10_\(\[RM\]\)\(\\d\+\)_N\(\\d\)P\(\\d\)\(\\d\)", cfg_name\)\n\s+if not npp_match:\n\s+continue\n\s+embed_type, dim, n_idx, p1_idx, p2_idx = npp_match\.groups\(\)'
new_regex = """npp_match = re.search(r"B11_(\\\\d+x)_([RM])(\\\\d+)_N1P21", cfg_name)
    if not npp_match:
        continue
    scale, embed_type, dim = npp_match.groups()
    n_idx, p1_idx, p2_idx = "1", "2", "1"
"""
content = re.sub(old_regex, new_regex, content)

# 3. Add scale to data row
content = content.replace('"config": cfg_name,', '"config": cfg_name,\n        "scale": scale,')

# 4. HTML Table headers (Find the noise header and insert scale before it)
content = content.replace(
    '<th data-col="noise">Noise</th>',
    '<th data-col="scale">Scale</th>\n                        <th data-col="noise">Noise</th>'
)

# 5. JS createRow (Find the noise_col injection and insert scale)
old_js = """            // Separate parameters cells
            if (columnState.noise_col) {"""
new_js = """            // Separate parameters cells
            {
                const td = document.createElement("td");
                td.className = "col-param";
                td.textContent = row.scale;
                tr.appendChild(td);
            }
            if (columnState.noise_col) {"""
content = content.replace(old_js, new_js)

b11_path.write_text(content)
b11_path.chmod(0o755)
print("generate_dashboard.py for B11 created successfully!")
