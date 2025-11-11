# Canopy Tree Height Estimation using Quantile Regression: Modeling and Evaluating Uncertainty in Remote Sensing

Preliminary work. Under review by AISTATS 2026. Do
not distribute.

## Training pipeline (Training Data required)

A training can be launched by running main.py.

## Validation (Validation Data required)

Using **notebooks/validation.ipynb**, result dataframes can be created. These include one row per labelled pixel in the 250 validation images with predictions and gedi labels. The resulting dataframes are stored in **results_dataframes**:
- **results_lang.pkl** (Results from model from Lang et. al.)
- **results_pauls.pkl** (Point-Estimator Results from model from Pauls et. al. (2025))
- **results_ours.pkl** (Results from our model with shift loss)
- **results_ours_wo_shift.pkl** (Results from our model without shift loss)
- **dem_values.pkl** (average slope at labelled pixels, generated using **notebooks/create_dem_values.ipynb** - downloaded dem patches needed)


## Evaluation 

Using the dataframes in **results_dataframes**, the results can be generated using 
- **notebooks/evaluation.ipynb** (No additional data needed)
- **notebooks/dem_analysis.ipynb** (No additional data needed)
- **notebooks/create_example_plots.ipynb** (Validation data needed)