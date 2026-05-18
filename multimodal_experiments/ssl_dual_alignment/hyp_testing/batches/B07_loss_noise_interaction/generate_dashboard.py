import re
import os
import shutil

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
status_file = os.path.join(script_dir, "STATUS.md")
output_file = os.path.join(script_dir, "VISUALIZER.html")
assets_dir = os.path.join(script_dir, "VISUALIZER_htmls")

if not os.path.exists(status_file):
    print(f"Error: {status_file} not found")
    exit(1)

if not os.path.exists(assets_dir):
    os.makedirs(assets_dir)

with open(status_file, "r") as f:
    lines = f.readlines()

data = []
for line in lines:
    clean_line = line.strip()
    if not clean_line or not clean_line.startswith("|"):
        continue
    
    cols = [c.strip() for c in clean_line.split("|")][1:-1]
    if len(cols) < 5:
        continue
    
    config = cols[0]
    if config == "Config" or "---" in config:
        continue
        
    params = cols[2]
    abs_link_cell = cols[5]
    
    # NPP extraction for sorting/mapping
    # B07_NPP[N][P1][P2]
    npp_match = re.search(r"B07_NPP(\d)(\d)(\d)", config)
    if not npp_match:
        continue
    
    n_idx, p1_idx, p2_idx = npp_match.groups()
    
    noise = "Unknown"
    prior = "Unknown"
    pred = "Unknown"
    
    # Parse params cell: "NoiseInfo, Pri:X, Pre:Y"
    # Example: "Asy:0.15/Ext:0.0, Pri:None, Pre:None"
    match = re.search(r"(.*), Pri:([^,]+), Pre:([^,]+)", params)
    if match:
        noise = match.group(1).strip()
        prior = match.group(2).strip()
        pred = match.group(3).strip()
    
    html_path = ""
    path_match = re.search(r"\[HTML\]\(([^)]+)\)", abs_link_cell)
    if path_match:
        html_path = path_match.group(1)
    
    local_rel_path = ""
    if html_path and os.path.exists(html_path):
        target_filename = f"{config}.html"
        target_path = os.path.join(assets_dir, target_filename)
        try:
            shutil.copy2(html_path, target_path)
            local_rel_path = f"VISUALIZER_htmls/{target_filename}"
        except Exception as e:
            print(f"Error copying {config}: {e}")
        
    data.append({
        "config": config,
        "n_idx": n_idx,
        "p1_idx": p1_idx,
        "p2_idx": p2_idx,
        "noise": noise,
        "prior": prior,
        "pred": pred,
        "local_path": local_rel_path
    })

# Define unique values and their display labels
# We want to sort by the NPP index to keep order logical (1-6, 0-2, 0-2)
noise_map = {}
prior_map = {}
pred_map = {}

for d in data:
    noise_map[d["noise"]] = d["n_idx"]
    prior_map[d["prior"]] = d["p1_idx"]
    pred_map[d["pred"]] = d["p2_idx"]

sorted_noises = sorted(noise_map.keys(), key=lambda x: noise_map[x])
sorted_priors = sorted(prior_map.keys(), key=lambda x: prior_map[x])
sorted_preds = sorted(pred_map.keys(), key=lambda x: pred_map[x])

def gen_options(items, key_map, default_val=None):
    opts = ['<option value="all">All</option>']
    for item in items:
        # Default is either first item or specifically NPP index 1 or 0
        is_default = False
        if default_val is not None:
            if key_map[item] == default_val:
                is_default = True
        elif item == items[0]:
            is_default = True
            
        selected = ' selected' if is_default else ''
        # Label with NPP index for clarity
        label = f"[{key_map[item]}] {item}"
        opts.append(f'<option value="{item}"{selected}>{label}</option>')
    return "".join(opts)

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>B07 Sweep Visualizer</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; margin: 20px; background: #fafafa; color: #24292f; }}
        h1 {{ font-size: 24px; border-bottom: 1px solid #d0d7de; padding-bottom: 10px; }}
        .filters {{ margin-bottom: 20px; padding: 15px; background: #fff; border: 1px solid #d0d7de; border-radius: 6px; position: sticky; top: 10px; z-index: 100; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
        .filter-group {{ display: inline-block; margin-right: 24px; }}
        label {{ font-weight: 600; margin-right: 8px; font-size: 14px; color: #57606a; }}
        select {{ padding: 6px 12px; border-radius: 6px; border: 1px solid #d0d7de; background-color: #f6f8fa; cursor: pointer; font-size: 13px; }}
        table {{ width: 100%; border-collapse: separate; border-spacing: 0; background: #fff; border: 1px solid #d0d7de; border-radius: 6px; overflow: hidden; }}
        th, td {{ border-bottom: 1px solid #d0d7de; padding: 12px; text-align: left; vertical-align: middle; }}
        th {{ background: #f6f8fa; font-weight: 600; border-bottom: 2px solid #d0d7de; font-size: 14px; }}
        .col-config {{ width: 1%; white-space: nowrap; font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace; font-size: 12px; font-weight: 600; background: #fdfdfd; }}
        .col-param {{ width: 1%; white-space: nowrap; font-size: 13px; color: #24292f; }}
        .embed-cell {{ padding: 0; width: 80%; }}
        .embed-container {{ width: 100%; height: 600px; resize: vertical; overflow: hidden; position: relative; border-left: 1px solid #d0d7de; }}
        iframe {{ width: 100%; height: 100%; border: none; }}
        .row {{ display: none; }}
        .placeholder {{ color: #57606a; font-style: italic; display: flex; align-items: center; justify-content: center; height: 100%; background: #f6f8fa; font-size: 14px; }}
        #stats {{ font-weight: 600; color: #0969da; margin-left: 12px; font-size: 14px; }}
        .config-label {{ font-size: 11px; color: #888; display: block; }}
    </style>
</head>
<body>
    <h1>B07 Sweep Visualizer</h1>
    
    <div class="filters">
        <div class="filter-group">
            <label for="filter-noise">Noise Regime [N]:</label>
            <select id="filter-noise" onchange="filter()">
                {gen_options(sorted_noises, noise_map, default_val='1')}
            </select>
        </div>
        <div class="filter-group">
            <label for="filter-prior">Prior Type [P1]:</label>
            <select id="filter-prior" onchange="filter()">
                {gen_options(sorted_priors, prior_map, default_val='0')}
            </select>
        </div>
        <div class="filter-group">
            <label for="filter-pred">Predictor Type [P2]:</label>
            <select id="filter-pred" onchange="filter()">
                {gen_options(sorted_preds, pred_map, default_val='0')}
            </select>
        </div>
        <span id="stats"></span>
    </div>

    <table id="main-table">
        <thead>
            <tr>
                <th class="col-config">Config (N P1 P2)</th>
                <th class="col-param">Noise</th>
                <th class="col-param">Prior</th>
                <th class="col-param">Pred</th>
                <th>Visualization</th>
            </tr>
        </thead>
        <tbody>
"""

for d in data:
    if d["local_path"]:
        embed = f'<div class="embed-container"><iframe data-src="{d["local_path"]}" src="about:blank" loading="lazy"></iframe></div>'
    else:
        embed = '<div class="embed-container"><div class="placeholder">No local HTML available for this run (Status might be TODO or link missing in MD)</div></div>'
        
    html_content += f"""
            <tr class="row" data-noise="{d["noise"]}" data-prior="{d["prior"]}" data-pred="{d["pred"]}">
                <td class="col-config">
                    {d["config"]}
                    <span class="config-label">N:{d["n_idx"]} P1:{d["p1_idx"]} P2:{d["p2_idx"]}</span>
                </td>
                <td class="col-param">{d["noise"]}</td>
                <td class="col-param">{d["prior"]}</td>
                <td class="col-param">{d["pred"]}</td>
                <td class="embed-cell">{embed}</td>
            </tr>
    """

html_content += """
        </tbody>
    </table>

    <script>
        function filter() {
            const noise = document.getElementById('filter-noise').value;
            const prior = document.getElementById('filter-prior').value;
            const pred = document.getElementById('filter-pred').value;
            
            const rows = document.querySelectorAll('.row');
            let visibleCount = 0;
            
            rows.forEach(row => {
                const rNoise = row.getAttribute('data-noise');
                const rPrior = row.getAttribute('data-prior');
                const rPred = row.getAttribute('data-pred');
                
                const matchNoise = (noise === 'all' || rNoise === noise);
                const matchPrior = (prior === 'all' || rPrior === prior);
                const matchPred = (pred === 'all' || rPred === pred);
                
                const iframe = row.querySelector('iframe');

                if (matchNoise && matchPrior && matchPred) {
                    row.style.display = 'table-row';
                    visibleCount++;
                    
                    if (iframe && iframe.hasAttribute('data-src')) {
                        const targetSrc = iframe.getAttribute('data-src');
                        if (iframe.src === 'about:blank' || !iframe.src.endsWith(targetSrc)) {
                            iframe.src = targetSrc;
                        }
                    }
                } else {
                    row.style.display = 'none';
                    if (iframe && iframe.src !== 'about:blank') {
                        iframe.src = 'about:blank';
                    }
                }
            });
            
            document.getElementById('stats').innerText = `Showing ${visibleCount} of ${rows.length} runs`;
        }
        
        if (document.readyState === 'complete' || document.readyState === 'interactive') {
            filter();
        } else {
            document.addEventListener('DOMContentLoaded', filter);
        }
    </script>
</body>
</html>
"""

with open(output_file, "w") as f:
    f.write(html_content)
print(f"Generated {output_file} with {len(data)} entries.")
