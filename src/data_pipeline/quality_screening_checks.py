import pandas as pd

q = pd.read_csv("data/manifests/quality_screening.csv")

band = q[(q.status == "fail")
         & q.person_detection_rate.between(0.6, 0.8)
         & (q.p10_frame_visibility >= 0.5)        # fails ONLY on detection
         & (q.height >= 240)]

for ex, g in band.groupby("exercise"):
    print(ex, len(g))
    print(g.sample(min(5, len(g)), random_state=0)[
        ["clip_id", "person_detection_rate", "p10_frame_visibility"]])
