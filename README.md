# Multivariate Visualization Dashboard
This interactive visualization dashboard enables exploration of high-dimensional data using dimensionality reduction techniques. We apply **PCA (Principal Component Analysis)** and **MDS (Multidimensional Scaling)** to a mammal sleep dataset, allowing intuitive understanding of multivariate relationships, patterns, and feature contributions.

---

## Dataset

The dataset `mammals.csv` contains 60 mammal species with the following 11 attributes:

| Column        | Description |
|---------------|-------------|
| `species`     | Name of the species |
| `body_wt`     | Body weight in kg |
| `brain_wt`    | Brain weight in kg |
| `non_dreaming`| Non-dreaming sleep (hours) |
| `dreaming`    | Dreaming sleep (hours) |
| `total_sleep` | Total sleep time (hours) |
| `life_span`   | Life expectancy (years) |
| `gestation`   | Gestation period (days) |
| `predation`   | Likelihood of being preyed on (1=low, 5=high) |
| `exposure`    | Exposure during sleep (1=protected, 5=exposed) |
| `danger`      | Danger level from predators (1=low, 5=high) |

Data source: [OpenIntro mammals dataset](https://www.openintro.org/data/index.php?data=mammals)

---

## Step 1: Preprocessing

- **Missing Values**:
  - Filled using K-Nearest Neighbors (KNN) imputation.
  - Rationale: Dataset has multiple correlated numeric features. KNN imputation leverages similarity between animals for plausible estimates.

- **Normalization**:
  - All 10 numerical features (excluding `species`) were normalized (mean=0, std=1).
  - Ensures each variable contributes equally to the analysis.

---

## Step 2: Principal Component Analysis (PCA)

### What was done:

- PCA was applied to the 10 normalized numerical variables using `scikit-learn`.
- All 10 principal components were computed.
- PCA variance explained is shown for each component.

### Interactive Features:

- Scatterplot of two user-selected principal components.
- Dropdowns: Choose any of the 10 components for x- and y-axis.
- Variance Display: Percent of variance explained by selected components is shown.
- Bar Chart:
  - Displays PCA loadings of the selected components.
  - X-axis: original variables; Y-axis: scaled loadings (by variance).

---

## Step 3: Multidimensional Scaling (MDS)

- MDS applied to the same 10-dimensional normalized data.
- 2D MDS projection created with `scikit-learn`.
- Side-by-side comparison with PCA scatterplot.

---

## Step 4: Interaction Features

### Enhanced Hover Information

- On hover, show all original attributes (including `species`) plus the PCA or MDS values.

### Dynamic Color Encoding

- Add dropdown to select one of the original variables for color mapping.
- Scatterplots (PCA & MDS) use a blue-to-red color scale for visual comparison.

**Extra**: Use this to validate loadings — e.g., if `body_wt` has high contribution in PC1, high values should appear red in the PC1 axis direction.

### Linked Selection
- Selecting points in either PCA or MDS plot (via box/lasso) highlights the same points in both plots.
- Implemented using `dcc.Store` for shared state between callbacks.

---

### Dependencies

Install required libraries with:

```bash
pip install dash pandas numpy scikit-learn plotly
```

### Run the App
```bash
python app.py
```

It then opens http://localhost:8000 in your browser.
