# Actin-Deformation Analysis

## Overview
This repository contains Python code to analyze time-lapse microscopy data of cells and extract quantitative information on local deformation. It identifies and compares local maxima in actin intensity (e.g., labeled cytoskeletal components) and mechanical deformation fields (e.g., from PIV data) to assess their spatial correlation.

## Features

- ✅ Load and process multi-timepoint, multi-stack data from `.pkl` files
- ✅ Apply adaptive mask inflation/deflation
- ✅ Detect local maxima in actin and deformation magnitude
- ✅ Match and quantify closest actin–deformation pairs
- ✅ Filter outliers using IQR-based criteria
- ✅ Visualize deformation magnitudes and distances via boxplots
- ✅ Overlay actin intensity, deformation vectors, and maxima for qualitative inspection

## Output

The code generates the following output files:

- `combined_deformation_closest_boxplot.png`: Boxplot of deformation magnitudes at actin-deformation peak pairs
- `combined_distance_closest_boxplot.png`: Boxplot of distances between actin and deformation peaks
- `summary_actin_deformation_overlay.png`: Visualization of peak alignment and deformation vectors for each dataset

## Directory Structure

```
/path/to/processed_pickle_data/
├── Dataset1/
│   └── processed_data.pkl
├── Dataset2/
│   └── processed_data.pkl
...

/path/to/output/plots/
```

## Configuration

Edit the script to define:

```python
path_pickle = '/path/to/processed_pickle_data'
plot_path = '/path/to/output/plots'
```

Key parameters:

| Parameter        | Description                                       |
|------------------|---------------------------------------------------|
| `inflation`      | Mask inflation factor in % (e.g. `110` = +10%)   |
| `radius`         | Neighborhood radius for peak detection           |
| `threshold_int`  | Intensity threshold for actin maxima             |
| `threshold_def`  | Threshold for deformation maxima                 |
| `maxDist`        | Maximum allowed distance (px) for pair matching  |

## Dependencies

Install required Python packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-image scipy
```

## Citation

If you use this code in scientific work, please cite the associated publication or acknowledge:

**Michael Riedl**

---

Feel free to open an issue if you have questions or suggestions.

## Citation

If you use this code in scientific work, please cite the associated publication or acknowledge:

**Michael Riedl**

Related publication:

**Global coordination of protrusive forces in migrating immune cells**  
Patricia Reis-Rodrigues, Nikola Canigova, Mario J. Avellaneda, Florian Gaertner, Kari Vaahtomeri, Michael Riedl, Jack Merrin, Robert Hauschild, Yoshinori Fukui, Alba Juanes Garcia, Michael Sixt  
DOI: [https://doi.org/10.1101/2024.07.26.605242](https://doi.org/10.1101/2024.07.26.605242)
