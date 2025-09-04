#%pip install -r requirements.txt
# --- Cell 1: Import libraries ---
# --- Cell 1: Import libraries ---
# --- Cell 1: Import libraries ---
import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx
import folium
import itertools
from geopy.distance import geodesic
import time
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import requests
import json

# Configure settings
ox.settings.use_cache = True
ox.settings.log_console = True
import json
import math
import numpy as np
from itertools import permutations
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler
from qiskit_optimization.converters import QuadraticProgramToQubo
from IPython.display import display, Math
# --- Cell 6b: Point-to-Point Best Route Helper Functions ---
def haversine(lat1, lon1, lat2, lon2, return_geometry=True):
    """
    Calculate the best driving route and distance between two points
    using OSRM (lat/lon coordinates).

    Parameters:
        lat1, lon1 : float - coordinates of point A
        lat2, lon2 : float - coordinates of point B
        return_geometry : bool - if True, also return route geometry

    Returns:
        distance_km : float
        duration_min : float
        geometry : list (only if return_geometry=True)
    """
    coords_str = f"{lon1},{lat1};{lon2},{lat2}"
    url = f"http://router.project-osrm.org/route/v1/driving/{coords_str}?overview=full&geometries=geojson"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        if data['code'] == 'Ok' and data['routes']:
            route = data['routes'][0]
            distance_km = route['distance'] / 1000  # meters → km
            duration_min = route['duration'] / 60   # seconds → minutes
            geometry = route['geometry']['coordinates']

            # Debug print
            print(f"Best route from ({lat1}, {lon1}) → ({lat2}, {lon2})")
            print(f"  Distance: {distance_km:.2f} km")
            print(f"  Duration: {duration_min:.2f} min")

            if return_geometry:
                return distance_km #, duration_min, geometry
            return distance_km, duration_min
        else:
            print("OSRM error: No route found")
            return None

    except requests.exceptions.RequestException as e:
        print(f"OSRM API error: {e}")
        return None

'''
def haversine(lat1, lon1, lat2, lon2):
    """Calculate the Haversine distance between two points."""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))
'''
# Load the data
with open("OptimizationProblemData.json", "r") as f:
    data = json.load(f)

hospital = data["locations"]["hospital"]["coordinates"]
patients = data["locations"]["patients"]
n_patients = len(patients)
max_stops = 3  # Maximum stops per trip

# Create a list of all locations (hospital first, then patients)
locations = [hospital] + [p["coordinates"] for p in patients]
n_locations = len(locations)

# Precompute distance matrix
dist_matrix = np.zeros((n_locations, n_locations))
for i in range(n_locations):
    for j in range(n_locations):
        if i != j:
            coord_i = locations[i]
            coord_j = locations[j]
            dist_matrix[i, j] = haversine(
                coord_i["latitude"], coord_i["longitude"],
                coord_j["latitude"], coord_j["longitude"]
            )
latex_str = r"\begin{bmatrix}" + \
            r"\\".join([" & ".join(map(str, row)) for row in dist_matrix]) + \
            r"\end{bmatrix}"

display(Math(latex_str))

# Create Quadratic Program with a more efficient formulation
qp = QuadraticProgram(name="Ambulance_Routing")

# Add binary variables: x_{i,t} - patient i is in trip t
for i in range(n_patients):
    for t in range(2):  # We need at most 2 trips for 5 patients
        qp.binary_var(name=f"x_{i}_{t}")

# Add constraint: Each patient visited exactly once
for i in range(n_patients):
    constraint_terms = {}
    for t in range(2):
        constraint_terms[f"x_{i}_{t}"] = 1
    qp.linear_constraint(
        linear=constraint_terms,
        sense="==",
        rhs=1,
        name=f"patient_{i}_visited_once"
    )

# Add constraint: Maximum patients per trip
for t in range(2):
    constraint_terms = {}
    for i in range(n_patients):
        constraint_terms[f"x_{i}_{t}"] = 1
    qp.linear_constraint(
        linear=constraint_terms,
        sense="<=",
        rhs=max_stops,
        name=f"trip_{t}_max_patients"
    )

# Build the objective function
linear_terms = {}
quadratic_terms = {}

# Distance terms for each trip
for t in range(2):
    # Hospital to first patient distance (approximation)
    for i in range(n_patients):
        linear_terms[f"x_{i}_{t}"] = dist_matrix[0, i+1] * 0.5  # Weighted

    # Distances between patients in the same trip
    for i in range(n_patients):
        for j in range(i+1, n_patients):
            quadratic_terms[(f"x_{i}_{t}", f"x_{j}_{t}")] = dist_matrix[i+1, j+1] * 0.3  # Weighted

# Convert to QUBO for better performance
converter = QuadraticProgramToQubo()
qubo = converter.convert(qp)

#print("Problem formulation complete")

#print(f"Variables: {qubo.get_num_vars()}")
#print(f"Constraints: {qubo.get_num_linear_constraints()}")

# Solve with QAOA using a simplified approach
optimizer = COBYLA(maxiter=50)
qaoa = QAOA(sampler=StatevectorSampler(), optimizer=optimizer, reps=10)
algorithm = MinimumEigenOptimizer(qaoa)

result = algorithm.solve(qubo)
print("\nOptimization result:")


# Interpret the solution
print("\nSuggested trips:")
trips = [[], []]  # Two trips
for i in range(n_patients):
    for t in range(2):
        var_name = f"x_{i}_{t}"
        if var_name in result.variables_dict and result.variables_dict[var_name] > 0.5:
            trips[t].append(patients[i]["id"])

# Calculate actual distances for each trip
def calculate_trip_distance(patient_ids):
    """Calculate the minimum distance for a trip visiting the given patients."""
    if not patient_ids:
        return 0, []

    # Get indices of patients in the distance matrix
    patient_indices = []
    for pid in patient_ids:
        for i, p in enumerate(patients):
            if p["id"] == pid:
                patient_indices.append(i+1)  # +1 because hospital is at index 0

    # Find the optimal order for this trip
    min_distance = float('inf')
    best_order = []

    # Try all permutations of patients to find the shortest path
    for order in permutations(patient_indices):
        distance = dist_matrix[0, order[0]]  # Hospital to first patient

        # Distances between patients
        for j in range(len(order)-1):
            distance += dist_matrix[order[j], order[j+1]]

        # Last patient back to hospital
        distance += dist_matrix[order[-1], 0]

        if distance < min_distance:
            min_distance = distance
            best_order = [patients[idx-1]["id"] for idx in order]

    return min_distance, best_order

total_distance = 0
for t in range(2):
    if trips[t]:
        distance, order = calculate_trip_distance(trips[t])
        total_distance += distance
        print(f"Trip {t+1}: Hospital -> {' -> '.join(order)} -> Hospital (Distance: {distance:.2f} km)")
    else:
        print(f"Trip {t+1}: Empty")

print(f"\nTotal distance: {total_distance:.2f} km")

# Print patient coordinates for reference
print("\nPatient coordinates:")
for i, patient in enumerate(patients):
    print(f"{patient['id']}: ({patient['coordinates']['latitude']:.6f}, {patient['coordinates']['longitude']:.6f})")
print(f"Hospital: ({hospital['latitude']:.6f}, {hospital['longitude']:.6f})")
