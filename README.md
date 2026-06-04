# High-Dimensional Feature Selection in Shallow Water Models

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest%20%7C%20Neural%20Networks-orange)
![Data Processing](https://img.shields.io/badge/Data-Xarray%20%7C%20Joblib-green)

A research project conducted at the Mathematical Institute for Machine Learning and Data Science (MIDS), KU Eichstätt-Ingolstadt, evaluating advanced dimensionality reduction techniques on large-scale spatial-temporal physics data.

📖 **[Read the Full Research Report (PDF)](docs/Report.pdf)**

## 📌 Project Overview
Predicting physical parameters in fluid dynamics (like Shallow Water Models) requires analyzing massive spatial-temporal grids. In this dataset, combinations of variables (velocity, height, rain) across grid points and time steps resulted in **750,000 potential features per simulation**. 

This project tackles the curse of dimensionality by comparing two distinct feature selection methodologies to find the most compact, predictive subset of features:
1. **The Information Imbalance Method** (Glielmo et al., 2022) - A distance-based metric evaluating the predictive information one space carries about another.
2. **Neural Network Backpropagation** - Extracting feature importance through the absolute weight gradients of a deeply regularized neural network.

## 🚀 Key Engineering Challenges & Solutions
Working with 5,000 simulations containing massive `.nc` arrays caused immediate Out-Of-Memory (OOM) failures on standard cloud hardware. I optimized the pipeline by:
* **Vectorizing Distance Matrices:** Replaced standard loops with `scipy.spatial.distance.cdist` and heavy `NumPy` vectorization.
* **Parallel Processing:** Implemented `joblib.Parallel` to distribute feature evaluation across all available virtual CPU cores.
* **Strategic Subsampling:** Capped initial feature evaluation spaces and utilized representative simulation subsets to compress calculation times from days to hours without losing statistical significance.

## 📊 Key Findings
The models were tasked with predicting three distinct targets: $\phi_c$, $h_{rain}$, and $\alpha$. The selected features (Top 5) were evaluated using a `RandomForestRegressor`.

* **Information Imbalance is Highly Efficient:** The Glielmo method maintained 76-99% of the full model's predictive power ($R^2$) while using only **0.0053%** of the original features.
* **Neural Networks Struggle with High Skewness:** While the NN approach showed moderate success on normally distributed targets, it completely failed on the highly right-skewed $\alpha$ parameter ($R^2$ dropped to 0.080), largely due to the rigid $L2$ regularization required to train the network.
* **Conclusion:** For physical interpretability and robust parameter estimation, information-theoretic distance measures vastly outperform standard deep learning gradient tracking.

## 🛠️ Repository Structure
* `/src`: Contains the optimized Python pipeline (`feature_pipeline.py`) used for parallel processing and model evaluation.
* `/docs`: Contains the final mathematical report (LaTeX) and oral presentation slides.
* `/data`: Directory meant for the raw simulation dataset.
* *Note: A `/results` directory containing JSON metrics and comparative plot visualizations is dynamically generated upon script execution.*

## 💻 Tech Stack
* **Data Processing:** `xarray`, `NumPy`, `Pandas`
* **Machine Learning:** `Scikit-Learn` (Random Forests, Scaling), `TensorFlow/Keras` (Neural Networks)
* **Optimization:** `Joblib` (Multiprocessing), `SciPy`

*Note: The raw `optimized_simulations_5k.nc` dataset is excluded from this repository due to size constraints. The pipeline expects this file in the root directory.*
