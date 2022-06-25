# Hyperopt search

Best metric search script based on hyperopt-sklearn.
HyperoptEstimator use 1-balanced_accuracy_score as loss func

## Usage

The script can be run without any parameters, but you can change them freely.


| Param name | Description   |  Default |
|---|---|---|
| score_func  | Sklearn metrics function. Evaluating the effectiveness before saving  |  balanced_accuracy_score | 
|  min_score | Min score to save model | 90 |  
|  classifier | Classifier function from hpsklearn package | any_classifier |  
|  evals |  Hyperopt max_evals param | 50  |  
|  scaler |  Apply StandardScaler before search | False  | 
|  pca | Apply PCA before search | False  | 
|  n_components | PCA n_components  | 0.98  | 
|  smote | Apply smote before  |  False | 
|  try_number | Number of executions HyperoptEstimator.fit() in loop  | 1  | 
|  hour | Minimum script execution time in hour  |  0 - disabled | 
|  trial_timeout | HyperoptEstimator trial_timeout param | 90 | 
|  log_level | Log level | 10 | 


## Conclusion
For the purposes of the project, the script was run with different parameters and combinations.
We get the best results when we enter raw data without preprocessing, for a better performance, we can increase it
evals and trial_timeout params