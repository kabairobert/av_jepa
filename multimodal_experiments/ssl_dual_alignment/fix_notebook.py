import json

path = '/gpfs/home3/rkabai/github/eb_jepa_private/multimodal_experiments/ssl_dual_alignment/Dataset_Tuner.ipynb'
with open(path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if 's_points = get_point_sizes(ds.param_values.numpy(), default_size=3.0, point_size_min=point_size_min, point_size_max=point_size_max)' in line:
                source[i] = "    s_points_a = get_point_sizes(ds.param_values.numpy(), default_size=3.0, point_size_min=point_size_min, point_size_max=point_size_max, point_types=ds.point_type_a.numpy())\n"
                source.insert(i+1, "    s_points_b = get_point_sizes(ds.param_values.numpy(), default_size=3.0, point_size_min=point_size_min, point_size_max=point_size_max, point_types=ds.point_type_b.numpy())\n")
                break

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if 's_points_clean = s_points[non_external]' in line:
                source[i] = "    s_points_latent_all = get_point_sizes(ds.param_values.numpy(), default_size=3.0, point_size_min=point_size_min, point_size_max=point_size_max, point_types=None)\n"
                source.insert(i+1, "    s_points_clean = s_points_latent_all[non_external] if isinstance(s_points_latent_all, np.ndarray) else s_points_latent_all\n")
                break

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if 'marker=dict(size=s_points, color=c_a' in line:
                source[i] = line.replace('size=s_points', 'size=s_points_a')
            if 'marker=dict(size=s_points, color=c_b' in line:
                source[i] = line.replace('size=s_points', 'size=s_points_b')

with open(path, 'w') as f:
    json.dump(nb, f, indent=1)
    f.write('\n')
