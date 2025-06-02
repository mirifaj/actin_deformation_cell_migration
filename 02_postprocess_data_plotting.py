# Import standard and scientific libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, binary_erosion, maximum_filter, minimum_filter, label, find_objects
from skimage.feature import peak_local_max
import seaborn as sns
import pickle

# --- CONFIGURATION ---
# Mask inflation factor (in %)
inflation = 110
# Local maximum detection radius
radius = 20
# Intensity and deformation thresholds
threshold_int = 800
threshold_def = 0.1
# Max distance (in px) allowed between actin and deformation peak to count as 'paired'
maxDist = 30

# Define data and output paths (customize these to your directory)
path_pickle = '/path/to/processed_pickle_data'
plot_path = '/path/to/output/plots'

# --- UTILITY FUNCTIONS ---

def read_pickle_file(pickle_path):
    """Read data from a .pkl file."""
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)

def local_maxima(data, radius, threshold):
    """Detect local maxima using morphological filters and thresholding."""
    data_max = maximum_filter(data, radius)
    maxima = (data == data_max)
    data_min = minimum_filter(data, radius)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0
    labeled, _ = label(maxima)
    slices = find_objects(labeled)
    x, y, values = [], [], []
    for dy, dx in slices:
        x_center = (dx.start + dx.stop - 1) / 2
        y_center = (dy.start + dy.stop - 1) / 2
        x.append(x_center)
        y.append(y_center)
        values.append(data_max[int(y_center), int(x_center)])
    return x, y, values

def calculate_distances(x1, y1, x2, y2):
    """Compute pairwise distances between two sets of points."""
    distances = {}
    for i in range(len(x1)):
        for j in range(len(x2)):
            p1 = (x1[i], y1[i])
            p2 = (x2[j], y2[j])
            distances[(p1, p2)] = np.linalg.norm(np.array(p1) - np.array(p2))
    return distances

def find_closest_pair(distances):
    """Return the pair of points with the smallest distance."""
    closest = min(distances, key=distances.get)
    return np.array([closest[0][0], closest[1][0]]), np.array([closest[0][1], closest[1][1]])

# --- MAIN ANALYSIS LOOP ---

# Gather all dataset names (folder names)
name_list = os.listdir(path_pickle)
all_deformation_closest = []
all_distance_closest = []
dataset_labels = []

for name in name_list:
    print(f"Processing {name}")
    try:
        # Load dataset
        data = read_pickle_file(os.path.join(path_pickle, name, 'processed_data.pkl'))
    except Exception as e:
        print(f"Failed to read {name}: {e}")
        continue

    timepoints = len(data) - 1
    stacks = len(data[0])
    max_datapoints = timepoints * stacks
    deformation_closest = []
    distance_closest = []

    # Loop through all timepoints and stacks
    for t in range(timepoints):
        for s in range(stacks):
            try:
                cell = data[t][s]['cell']
                def_x = data[t][s]['deform_x']
                def_y = data[t][s]['deform_y']
                mask = (data[t][s]['mask'] > 128).astype(np.uint8)
                def_mag = np.sqrt(def_x**2 + def_y**2)

                if np.count_nonzero(mask) == 0:
                    continue

                # Inflate or deflate the mask to standardize region size
                mask_proc = mask.copy()
                target_ratio = inflation / 100.0
                original_size = np.count_nonzero(mask_proc)

                while True:
                    new_mask = binary_dilation(mask_proc) if inflation > 100 else binary_erosion(mask_proc)
                    new_size = np.count_nonzero(new_mask)
                    if new_size == mask_proc.sum() or new_size == 0:
                        break
                    if ((inflation > 100 and new_size / original_size >= target_ratio) or
                        (inflation < 100 and new_size / original_size <= target_ratio)):
                        mask_proc = new_mask.astype(np.uint8)
                        break
                    mask_proc = new_mask.astype(np.uint8)

                # Apply mask to cell intensity and deformation magnitude
                roi_cell = cell * mask_proc
                roi_def = def_mag * mask_proc

                # Find local maxima in actin and deformation channels
                x_int, y_int, val_int = local_maxima(roi_cell, radius, threshold_int)
                x_def, y_def, val_def = local_maxima(roi_def, radius, threshold_def)

                if not x_int or not x_def:
                    continue

                # Calculate all distances and identify the closest pair
                distances = calculate_distances(x_int, y_int, x_def, y_def)
                _, y_coords = find_closest_pair(distances)
                idx_def = np.where(y_def == y_coords[1])[0][0]

                # Only keep if the closest match is within maxDist
                min_dist = min(distances.values())
                if min_dist <= maxDist:
                    deformation_closest.append(val_def[idx_def])
                    distance_closest.append(min_dist)

            except Exception as e:
                print(f"Error processing time {t}, stack {s}: {e}")
                continue

    # Outlier filtering (IQR-based)
    if len(deformation_closest) > 0:
        deformation_closest = np.array(deformation_closest)
        distance_closest = np.array(distance_closest)

        print(f"{name}: expected ≤ {max_datapoints}, actual = {len(deformation_closest)}")

        q1, q3 = np.percentile(deformation_closest, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        filtered_def = deformation_closest[(deformation_closest >= lower) & (deformation_closest <= upper)]

        q1_d, q3_d = np.percentile(distance_closest, [25, 75])
        iqr_d = q3_d - q1_d
        lower_d, upper_d = q1_d - 1.5 * iqr_d, q3_d + 1.5 * iqr_d
        filtered_dist = distance_closest[(distance_closest >= lower_d) & (distance_closest <= upper_d)]

        all_deformation_closest.append(filtered_def)
        all_distance_closest.append(filtered_dist)
        dataset_labels.append(name)

# --- BOXPLOT: DEFORMATION MAGNITUDE ---
if all_deformation_closest:
    fig, ax = plt.subplots(figsize=(1.5 * len(dataset_labels), 6))
    df = pd.DataFrame(all_deformation_closest).T
    sns.boxplot(data=df, showfliers=False, ax=ax, palette='light:#66b3ff')
    sns.stripplot(data=df, color='black', size=4, jitter=True, ax=ax)
    ax.set_xticklabels(dataset_labels, rotation=45, ha='right')
    ax.set_ylabel('Deformation magnitude')
    ax.set_title('Max Deformation @ Closest Actin-Deformation Pair')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, "combined_deformation_closest_boxplot.png"), dpi=300)
    plt.close()

# --- BOXPLOT: DISTANCE BETWEEN PEAKS ---
if all_distance_closest:
    fig, ax = plt.subplots(figsize=(1.5 * len(dataset_labels), 6))
    df_dist = pd.DataFrame(all_distance_closest).T
    sns.boxplot(data=df_dist, showfliers=False, ax=ax, palette='light:#ff9999')
    sns.stripplot(data=df_dist, color='black', size=4, jitter=True, ax=ax)
    ax.set_xticklabels(dataset_labels, rotation=45, ha='right')
    ax.set_ylabel('Distance (px)')
    ax.set_title('Actin-Deformation Distance (Closest Pair)')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, "combined_distance_closest_boxplot.png"), dpi=300)
    plt.close()

# --- VISUALIZE BEST STACK (PER DATASET) ---
fig, axs = plt.subplots(1, len(name_list), figsize=(6 * len(name_list), 6))
if len(name_list) == 1:
    axs = [axs]  # Ensure iterable even with single subplot

for i, name in enumerate(name_list):
    try:
        data = read_pickle_file(os.path.join(path_pickle, name, 'processed_data.pkl'))
    except Exception as e:
        print(f"Skipping {name} for visualization: {e}")
        continue

    # Find the stack with the strongest masked actin signal
    timepoints = len(data) - 1
    stacks = len(data[0])
    best_mean = -np.inf
    best_t, best_s = 0, 0

    for t in range(timepoints):
        for s in range(stacks):
            cell = data[t][s]['cell']
            mask = (data[t][s]['mask'] > 128).astype(np.uint8)
            masked_cell = cell * mask
            mean_val = masked_cell[masked_cell > 0].mean() if np.any(masked_cell > 0) else 0
            if mean_val > best_mean:
                best_mean = mean_val
                best_t, best_s = t, s

    # Load corresponding data
    d = data[best_t][best_s]
    cell = d['cell']
    def_x = d['deform_x']
    def_y = d['deform_y']
    def_mag = np.sqrt(def_x**2 + def_y**2)
    mask = (d['mask'] > 128).astype(np.uint8)

    roi_cell = cell * mask
    roi_def_x = def_x * mask
    roi_def_y = def_y * mask
    roi_def_mag = def_mag * mask

    # Find maxima in both channels
    x_int, y_int, _ = local_maxima(roi_cell, radius, threshold_int)
    x_def, y_def, _ = local_maxima(roi_def_mag, radius, threshold_def)

    ax = axs[i]
    ax.contourf(roi_cell, cmap='magma', levels=100)

    # Normalize vectors and overlay as quiver plot
    Y, X = np.mgrid[0:roi_def_x.shape[0], 0:roi_def_x.shape[1]]
    mag = np.sqrt(roi_def_x**2 + roi_def_y**2)
    with np.errstate(invalid='ignore', divide='ignore'):
        u = np.nan_to_num(roi_def_x / mag)
        v = np.nan_to_num(roi_def_y / mag)

    ax.quiver(X[::5, ::5], Y[::5, ::5], u[::5, ::5], v[::5, ::5], mag[::5, ::5], cmap='viridis', scale=50)
    ax.scatter(x_int, y_int, color='orange', s=30, label='Actin max')
    ax.scatter(x_def, y_def, color='cyan', s=30, label='Def. max')
    ax.set_title(name)
    ax.set_aspect('equal')
    ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(plot_path, "summary_actin_deformation_overlay.png"), dpi=300)
plt.close()
