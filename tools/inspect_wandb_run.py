import sys
import json
from collections import defaultdict

run_id = "o17io1li"
# Candidate prefixes to try (adjust if your entity/project differs)
candidates = [
    f"robertkabai-um/eb_jepa/{run_id}",
    f"robertkabai-um/video_jepa/{run_id}",
    f"robertkabai-um/eb_jepa/{run_id}",
]

try:
    import wandb
except Exception as e:
    print("Failed to import wandb:", e, file=sys.stderr)
    sys.exit(3)

api = wandb.Api()
found = None
errors = []
for path in candidates:
    try:
        run = api.run(path)
        found = (path, run)
        break
    except Exception as e:
        errors.append((path, str(e)))

if not found:
    # Try to search the "eb_jepa" project for a run with matching id
    try:
        proj = "robertkabai-um/eb_jepa"
        print(f"Trying project search in {proj}...")
        runs = api.runs(proj, filters={})
        for r in runs:
            if r.id == run_id or r.name == run_id or getattr(r, 'short_id', None) == run_id:
                found = (f"{proj}/{r.id}", r)
                break
    except Exception as e:
        errors.append(("search_project", str(e)))

if not found:
    print("Could not find run with candidates. Errors:\n", file=sys.stderr)
    for p, e in errors:
        print(p, "->", e, file=sys.stderr)
    sys.exit(2)

path, run = found
print("Found run:", path)
print("Run name:", getattr(run, 'name', None))
print("Run id:", run.id)
print("Summary keys:", list(run.summary.keys())[:20])

# Scan history for geometry_viz keys
counts = defaultdict(int)
entries = []
for rec in run.scan_history():
    step = rec.get('_step')
    for k, v in rec.items():
        if isinstance(k, str) and k.startswith('geometry_viz/') and v is not None:
            counts[k] += 1
            entries.append((step, k))

print('\nGeometry media counts:')
print(json.dumps(counts, indent=2))

print('\nSample media entries (step, key)')
for e in entries[:50]:
    print(e)

# Also print all unique steps that have any geometry media
steps_with_media = sorted(set(s for s, _k in entries if s is not None))
print('\nSteps with geometry media:', steps_with_media)

print('\nDone')
