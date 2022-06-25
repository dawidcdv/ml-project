# REPORT


#### Table of contents
* [Inspecting and cleanig data](#inspecting-and-cleanig-data)
* [Metrics](#metrics)
* [Baseline](#baseline)
* [Datasplit](#datasplit)
* [Classification](#classification)

## Inspecting and cleanig data

    - train_data have 3750 rows and 10000 columns
    - test_data have 1250 rows and 10000 columns
    - train_labels have 3750 rows and 1 column, there is 3375 in 1 class and 375 in -1 class
    - there is no NaN in all datasets
    - no corelation higher than 0.95
    - no low variance columns (treshold 0.5)
    - no outliers with 5 sigma
    
### TSNE PLOT
![TSNE](https://github.com/dawidcdv/ml-project/blob/master/reports/figures/tsne_scatter.png?raw=true)

### PCA + TSNE PLOT
![TSNEPCA](https://github.com/dawidcdv/ml-project/blob/master/reports/figures/tsne_pca_scatter.png?raw=true)

### PCA n_components for variance 0.95 is 3074, and for variance 0.99 is 3558 
![Pca n_components](https://github.com/dawidcdv/ml-project/blob/master/reports/figures/pca_n_components_scatter.png?raw=true)
    
        
## Metrics
    The most reliable choice for such an unbalanced dataset would be ballanced accuracy 
    We tested the roc auc as well, f0.5 and f2 score were worse .
    Classification report and confusion matrix allowed us to ultimately evaluate the effectiveness of the model
    
    
## Baseline
    DummyClassifier with its basic parameters indicated only one class
![Baseline cm](https://github.com/dawidcdv/ml-project/blob/master/reports/figures/dummy_cm.jpg?raw=true)
    
    
    
## Datasplit
    The data was split by train test split, while each model achieved a certain effectiveness
    (approximately 90% balanced accuracy score) and was considered by us to check for potential problems with
    overfitting, which was verified by cross valid split in 5-10 folds. We use 30% of the data for testing
    
    
## Classification
    In order to find the best classifier, we used the class HyperoptEstimator from the hpsklearn package.
    One of the best mdoels was the estimator:
    SVC with following parameters:
        C = 1.0752079389673375
        coef0 = 0.043809020064986326
        degree = 2
        Kernel = polly
        gamma = auto
        tol = 9.287881080560548e-05
        random_state = 4
        
    It allowed to obtain the result cross val score: 0.9390
    
![svc cm](https://github.com/dawidcdv/ml-project/blob/master/reports/figures/svc_cm.jpg?raw=true)
