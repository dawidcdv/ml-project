# RAPORT

## Inspecting and cleanig data

    - train_data have 3750 rows and 10000 columns
    - test_data have 1250 rows and 10000 columns
    - train_labels have 3750 rows and 1 column, there is 3375 in 1 class and 375 in -1 class
    - there is no NaN in all datasets
    - no corelation higher than 0.95
    - no low variance columns (treshold 0.5)
    - no outliers
    - best PCA n_components = 3074