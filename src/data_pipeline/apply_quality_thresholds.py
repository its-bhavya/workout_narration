import pandas as pd

q = pd.read_csv("data/manifests/quality_screening.csv")
q["split"] = q["split"].replace({"valid": "val"})    # normalise RepCount "valid" -> "val"
u = q[q.status != "fail"]
print(pd.crosstab([u.split], u.exercise))