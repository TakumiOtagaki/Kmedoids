# label: /large/otgk/app/kmedoids-parallel/v1.0.2/test/test300.label.csv
# medoid: /large/otgk/app/kmedoids-parallel/v1.0.2/test/test300.medoid.csv
# coordinates: /large/otgk/app/kmedoids-parallel/v1.0.2/test/test_points_n300r0.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


labels = pd.read_csv("/large/otgk/app/kmedoids-parallel/v1.0.2/test/test300.label.csv", header=None).values.flatten()
medoids = pd.read_csv("/large/otgk/app/kmedoids-parallel/v1.0.2/test/test300.medoid.csv", header=None).values.flatten()
points = pd.read_csv("/large/otgk/app/kmedoids-parallel/v1.0.2/test/test_points_n300r0.csv", header=None).values

print(points.shape)
print("labels", labels)
print("medoids", medoids)
# plot (coloring)
# print(points)
plt.figure()
for i in range(len(medoids)):
    plt.scatter(points[labels==i, 0], points[labels==i, 1], label=f"cluster{i}")
plt.scatter(points[medoids, 0], points[medoids, 1], marker="*", c="black", label="medoid")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("kmedoids-parallel")
plt.savefig("/large/otgk/app/kmedoids-parallel/v1.0.2/test/test300_kmresult.png")
plt.close()
