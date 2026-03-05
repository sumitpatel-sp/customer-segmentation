import requests, json

API = "https://customer-segmentation-2-166m.onrender.com/predict"

tests = [
    {"label": "Low income + Low spending",   "Gender": 0, "Age": 30, "Income": 18.0, "Spending": 20},
    {"label": "Low income + High spending",  "Gender": 0, "Age": 25, "Income": 18.0, "Spending": 75},
    {"label": "High income + Low spending",  "Gender": 0, "Age": 40, "Income": 90.0, "Spending": 20},
    {"label": "High income + High spending", "Gender": 0, "Age": 35, "Income": 90.0, "Spending": 80},
]

lines = []
for t in tests:
    label = t.pop("label")
    r = requests.post(API, json=t, timeout=20)
    d = r.json()
    lines.append(f"{label}")
    lines.append(f"  -> Cluster {d['Cluster']} -> {d['segment_name']}")
    lines.append("")

output = "\n".join(lines)
print(output)
with open("debug_result.txt", "w") as f:
    f.write(output)
