
import os
import json
import csv
from collections import defaultdict

base_dir = "_outputs"
output_csv = "3DGStream+ours.csv"

scene_metrics = defaultdict(dict)
all_metrics = defaultdict(list)

for scene in os.listdir(base_dir):
    scene_dir = os.path.join(base_dir, scene)
    json_path = os.path.join(scene_dir, "results.json")

    if not os.path.isfile(json_path):
        continue

    with open(json_path, 'r') as f:
        data = json.load(f)

    scene_metrics[scene] = data

    for k, v in data.items():
        all_metrics[k].append(v)

metric_names = list(next(iter(scene_metrics.values())).keys())

with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Scene"] + metric_names)

    for scene, metrics in sorted(scene_metrics.items()):
        row = [scene] + [metrics[k] for k in metric_names]
        writer.writerow(row)

    avg_row = ["Avg"] + [sum(all_metrics[k]) / len(all_metrics[k]) for k in metric_names]
    writer.writerow(avg_row)
