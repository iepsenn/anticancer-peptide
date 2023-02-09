## You can add/remove/change these hyper-parameters as you wish.

GNB_parameter_space = {
    'var_smoothing': [1e-9, 1e-6, 1e-3, 1e-1]}

KNN_parameter_space = {
    'n_neighbors': list(range(2,11)),
	'weights': ['uniform','distance'],
	'metric': ['euclidean', 'manhattan']}

SVM_parameter_space = {
        'C': [0.001, 0.01, 0.1, 1, 10],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'max_iter': [20000],
        'degree': (1, 15),  # integer valued parameter
        'kernel': ['linear', 'sigmoid', 'rbf'],
        'class_weight': [{0: 1., 1: 2.}]}

RF_parameter_space = {
    'n_estimators': [10,50,75,100,150],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']}

XGB_parameter_space = {
    "colsample_bytree": [i/10.0 for i in range(1, 3)],
    "gamma": [i/10.0 for i in range(3)],
    "learning_rate": [0.01, 0.1, 0.5, 1], # default 0.1
    "max_depth": [2, 3, 4, 5, 6, 8, 10], # default 3
    "n_estimators": [10, 50, 100, 150, 200, 500], # default 100
    "subsample": [i/10. for i in range(7,11)],
	"n_jobs": [1],
	"verbosity": [0]}
