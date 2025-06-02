# Import required libraries
import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pandas as pd
import pickle


# Base path to the dataset (update to your local or remote directory)
path_datasets = '/path/to/dataset'

# --- Utility Functions ---

def get_folder_names(directory_path):
    """Return a list of folder names in a given directory."""
    folder_names = []
    try:
        for item in os.listdir(directory_path):
            if os.path.isdir(os.path.join(directory_path, item)):
                folder_names.append(item)
    except FileNotFoundError:
        print(f"Error: The directory '{directory_path}' does not exist.")
    return folder_names

def get_tif_file(path):
    """Return the first .tif file found in the given directory."""
    tif_files = [f for f in os.listdir(path) if f.endswith('.tif')]
    if not tif_files:
        raise FileNotFoundError(f"No .tif files found in {path}")
    return os.path.join(path, tif_files[0])

def split_tif_channels(tif_path):
    """Split a 5D TIFF into two image channels."""
    image_data = tiff.imread(tif_path)  # Shape: (T, Z, C, Y, X)
    if len(image_data.shape) == 5:
        timepoints, stacks, channels, height, width = image_data.shape
        channel_1 = image_data[:, :, 0, :, :]
        channel_2 = image_data[:, :, 1, :, :]
    else:
        raise ValueError("Unexpected .tif format. Expected (time, stacks, channels, height, width)")
    return channel_1, channel_2, timepoints, stacks

def read_csv_file(file_path):
    """Read CSV file without headers and assign expected column names."""
    df = pd.read_csv(file_path, header=None)
    if df.shape[1] < 4:
        raise ValueError(f"CSV file {file_path} has fewer than 4 columns.")
    df.columns = ["X", "Y", "Vx", "Vy"]
    return df

def sorted_numerically(folder_list):
    """Sort a list of folder names containing numeric suffixes (e.g., 'Slice3')."""
    return sorted(folder_list, key=lambda x: int(x.replace("Slice", "")))

def resample_image_to_custom_grid(image, grid_x_values, grid_y_values):
    """Interpolate image data onto a defined (x, y) grid."""
    if image.ndim == 2:
        green_channel = image
    else:
        green_channel = image[:, :, 1]  # Use green channel if RGB

    y = np.linspace(0, image.shape[0] - 1, image.shape[0])
    x = np.linspace(0, image.shape[1] - 1, image.shape[1])
    original_grid_x, original_grid_y = np.meshgrid(x, y, indexing='xy')

    points = np.c_[original_grid_x.ravel(), original_grid_y.ravel()]
    values = green_channel.ravel()
    new_grid_x, new_grid_y = np.meshgrid(grid_x_values, grid_y_values, indexing='xy')
    green_channel_resampled = griddata(points, values, (new_grid_x, new_grid_y), method='cubic', fill_value=np.nan)

    green_channel_resampled = np.nan_to_num(green_channel_resampled, nan=np.nanmean(green_channel_resampled))
    return green_channel_resampled

def plot_results(img_interp_cell, img_interp_coll, img_interp_mask, img_interp_cell_r, img_interp_coll_r, img_interp_mask_r, Vx_interpolated, Vy_interpolated, new_grid_x_values, new_grid_y_values):
    """Visualize resampled images and interpolated deformation fields."""
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    axs[0, 0].imshow(img_interp_cell, cmap='gray'); axs[0, 0].set_title("Rescaled Cell Image")
    axs[0, 1].imshow(img_interp_coll, cmap='gray'); axs[0, 1].set_title("Rescaled Collagen Image")
    axs[0, 2].imshow(img_interp_mask, cmap='gray'); axs[0, 2].set_title("Rescaled Mask Image")

    cf1 = axs[1, 0].contourf(new_grid_x_values, new_grid_y_values, img_interp_cell_r, cmap='jet')
    fig.colorbar(cf1, ax=axs[1, 0]); axs[1, 0].set_title("Vx Deformation Field")

    cf2 = axs[1, 1].contourf(new_grid_x_values, new_grid_y_values, img_interp_coll_r, cmap='jet')
    fig.colorbar(cf2, ax=axs[1, 1]); axs[1, 1].set_title("Vy Deformation Field")

    V_magnitude = np.sqrt(Vx_interpolated**2 + Vy_interpolated**2)
    cf3 = axs[1, 2].contourf(new_grid_x_values, new_grid_y_values, V_magnitude, cmap='jet')
    fig.colorbar(cf3, ax=axs[1, 2]); axs[1, 2].set_title("|V| Deformation Magnitude")

    plt.tight_layout(); plt.show()

# --- Main Processing ---
folder_datasets = get_folder_names(path_datasets)


for folder in np.arange(len(folder_datasets)):
    environment = folder_datasets[folder]
    folder_names = get_folder_names(os.path.join(path_datasets, environment))

    tif_path = get_tif_file(os.path.join(path_datasets, environment))
    try:
        cell_channel, collagen_channel, N_timepoints, N_stacks = split_tif_channels(tif_path)
    except FileNotFoundError as e:
        print(e)
        exit()

    path_common = os.path.join(path_datasets, environment)
    path_mask = os.path.join(path_common, 'CellMask')
    path_def = os.path.join(path_common, 'PIV')

    sorted_slices = sorted_numerically(os.listdir(path_mask))

    data = {}
    stacks = np.arange(N_stacks)
    timepoints = np.arange(N_timepoints - 1)

    for time in timepoints:
        data[time] = {}
        for stack in stacks:
            file_cell = cell_channel[time, stack, :, :]
            file_coll = collagen_channel[time, stack, :, :]
            file_path_mask = get_tif_file(os.path.join(path_mask, sorted_slices[stack]))
            file_mask = tiff.imread(file_path_mask)

            csv_files = [f for f in os.listdir(os.path.join(path_def, sorted_slices[stack])) if f.endswith('.csv')]
            data_def = os.path.join(path_def, sorted_slices[stack], csv_files[time])
            df = read_csv_file(data_def)

            grid_x_values = df.X.values[~np.isnan(df.X.values)]
            grid_y_values = df.Y.values[~np.isnan(df.Y.values)]
            grid_y, grid_x = np.meshgrid(grid_y_values, grid_x_values)
            x_valid = grid_x.flatten()
            y_valid = grid_y.flatten()
            Vx = df.Vx.values
            Vy = df.Vy.values

            points = np.vstack((x_valid, y_valid)).T
            new_grid_x_values = np.linspace(np.nanmin(x_valid), np.nanmax(x_valid), 400)
            new_grid_y_values = np.linspace(np.nanmin(y_valid), np.nanmax(y_valid), 400)
            new_grid_x, new_grid_y = np.meshgrid(new_grid_x_values, new_grid_y_values)

            Vx_interpolated = griddata(points, Vx, (new_grid_x, new_grid_y), method='cubic')
            Vy_interpolated = griddata(points, Vy, (new_grid_x, new_grid_y), method='cubic')

            img_interp_cell = resample_image_to_custom_grid(file_cell, new_grid_x_values, new_grid_y_values)
            img_interp_coll = resample_image_to_custom_grid(file_coll, new_grid_x_values, new_grid_y_values)
            img_interp_mask = resample_image_to_custom_grid(file_mask, new_grid_x_values, new_grid_y_values)

            img_interp_cell_r = np.flipud(img_interp_cell)
            img_interp_coll_r = np.flipud(img_interp_coll)
            img_interp_mask_r = np.flipud(img_interp_mask)

            sample_field_data = {
                'cell': img_interp_cell_r,
                'collagen': img_interp_coll_r,
                'mask': img_interp_mask_r,
                'deform_x': Vx_interpolated,
                'deform_y': Vy_interpolated,
            }

            data[time][stack] = sample_field_data
            print(f'time: {time}, stack: {stack}')

    # Save processed dataset
    output_path = os.path.join(path_datasets, environment, 'processed_data.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
