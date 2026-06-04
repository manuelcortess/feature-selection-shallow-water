# Dataset Information

The raw data file used for this project (`optimized_simulations_5k.nc`) is extremely large and contains 750,000 spatial-temporal features across 5,000 fluid dynamics simulations. 

Due to GitHub's file size limits, it is excluded from this repository via `.gitignore`. 

To run the pipeline locally, ensure the `.nc` file is placed directly into this folder, or adjust the `DATA_PATH` in `src/feature_pipeline.py`.