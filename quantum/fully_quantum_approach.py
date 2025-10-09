#%pip install -r requirements.txt


import json
import numpy as np
from itertools import permutations, product
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler
from IPython.display import display, Math

def metric(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

with open("OptimizationProblemData.json", "r") as f:
    data = json.load(f)

hospital = data["locations"]["hospital"]["coordinates"]
patients = data["locations"]["patients"]
n_patients = len(patients)
max_stops = 3  # max patients per trip
n_trips = 2

locations = [hospital] + [p["coordinates"] for p in patients]
n_locations = len(locations)

# Distance matrix
dist_matrix = np.zeros((n_locations, n_locations))
for i in range(n_locations):
    for j in range(n_locations):
        if i != j:
            dist_matrix[i,j] = metric(
                locations[i]["latitude"], locations[i]["longitude"],
                locations[j]["latitude"], locations[j]["longitude"]
            )

latex_str = r"\begin{bmatrix}" + \
            r"\\".join([" & ".join(map(str,row)) for row in dist_matrix]) + \
            r"\end{bmatrix}"
# uncomment iff u r using notebooks
#display(Math(latex_str))


qp = QuadraticProgram(name="Ambulance_Routing")

positions = max_stops
for i in range(n_patients):
    for t in range(n_trips):
        for p in range(positions):
            qp.binary_var(name=f"x_{i}_{t}_{p}")

# Constraints
# 1) Each patient appears exactly once
for i in range(n_patients):
    linear_terms = {}
    for t in range(n_trips):
        for p in range(positions):
            linear_terms[f"x_{i}_{t}_{p}"] = 1
    qp.linear_constraint(linear=linear_terms, sense="==", rhs=1, name=f"patient_{i}_once")

# 2) Each position has at most one patient
for t in range(n_trips):
    for p in range(positions):
        linear_terms = {}
        for i in range(n_patients):
            linear_terms[f"x_{i}_{t}_{p}"] = 1
        qp.linear_constraint(linear=linear_terms, sense="<=", rhs=1, name=f"trip_{t}_pos_{p}_one")


linear_terms = {}
quadratic_terms = {}

for t in range(n_trips):
    for p in range(positions):
        for i in range(n_patients):
            var_i = f"x_{i}_{t}_{p}"
            # Distance from hospital to first patient
            if p == 0:
                linear_terms[var_i] = dist_matrix[0, i+1]
            # Distance from patient to hospital if last position (optional for linear)
            if p == positions-1:
                linear_terms[var_i] = linear_terms.get(var_i,0) + dist_matrix[i+1,0]

            # Distances between consecutive positions
            if p < positions-1:
                for j in range(n_patients):
                    var_j = f"x_{j}_{t}_{p+1}"
                    quadratic_terms[(var_i,var_j)] = dist_matrix[i+1,j+1]

# Add objective to QP
qp.minimize(linear=linear_terms, quadratic=quadratic_terms)

# Convert to QUBO
converter = QuadraticProgramToQubo()
qubo = converter.convert(qp)

optimizer = COBYLA(maxiter=100)
qaoa = QAOA(sampler=StatevectorSampler(), optimizer=optimizer, reps=3)
algorithm = MinimumEigenOptimizer(qaoa)

result = algorithm.solve(qubo)
print("\nOptimization result:", result)

# Decode solution
trips = [[] for _ in range(n_trips)]
for i in range(n_patients):
    for t in range(n_trips):
        for p in range(positions):
            var_name = f"x_{i}_{t}_{p}"
            if result.variables_dict.get(var_name,0) > 0.5:
                trips[t].append((p, patients[i]["id"]))

# Sort by position
for t in range(n_trips):
    trips[t].sort()
    trips[t] = [pid for pos,pid in trips[t]]

# Compute distances
def trip_distance(trip_ids):
    if not trip_ids: return 0
    idxs = [i+1 for i,p in enumerate(patients) if p["id"] in trip_ids]
    dist = dist_matrix[0, idxs[0]]  # hospital -> first
    for i in range(len(idxs)-1):
        dist += dist_matrix[idxs[i], idxs[i+1]]
    dist += dist_matrix[idxs[-1], 0]  # last -> hospital
    return dist

total_distance = 0
for t in range(n_trips):
    dist = trip_distance(trips[t])
    total_distance += dist
    print(f"Trip {t+1}: Hospital -> {' -> '.join(trips[t])} -> Hospital (Distance: {dist:.2f} km)")

print(f"\nTotal distance: {total_distance:.2f} km")

