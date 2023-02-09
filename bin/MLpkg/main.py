'''
This is a ready to use package consisting of several supervised machine learning
algorithms with a predefined parameter-grid (which then you can update its values)
to fine tune these models on your data. You can calculate some features from your
peptides data set using our feature extraction tool to feed these machines.

Authors: Saman Behrouzi; Fereshteh Fallah;
'''



import pandas as pd
import numpy as np
import sys, getopt, os
from sklearn.feature_selection import SelectKBest, chi2, RFE, SelectFromModel, f_classif
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error
from imblearn.metrics import specificity_score
from sklearn.metrics import matthews_corrcoef
from sklearn.utils import parallel_backend
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import tools
import hparams


def print_performance_results(results, model, model_name):
    print("* * * * * \n\nBest parameters: \n\n {}\n\n* * * * * ".format(model.named_steps[model_name].best_params_))
    print("\n - - Train performance results - -\n")
    print('f1_score, mean: {:0.3}'.format(results['test_f1_score'].mean()))
    print('f1_score, std: {:0.3}'.format(results['test_f1_score'].std()))
    print('roc_auc_score, mean: {:0.3}'.format(results['test_roc_auc_score'].mean()))
    print('roc_auc_score, std: {:0.3}'.format(results['test_roc_auc_score'].std()))
    print('accuracy, mean: {:0.3}'.format(results['test_accuracy'].mean()))
    print('accuracy, std: {:0.3}'.format(results['test_accuracy'].std()))
    print('precision, mean: {:0.3}'.format(results['test_precision'].mean()))
    print('precision, std: {:0.3}'.format(results['test_precision'].std()))
    print('recall, mean: {:0.3}'.format(results['test_recall'].mean()))
    print('recall, std: {:0.3}'.format(results['test_recall'].std()))
    print('specificity, mean: {:0.3}'.format(results['test_specificity'].mean()))
    print('specificity, std: {:0.3}'.format(results['test_specificity'].std()))
    print('mcc, mean: {:0.3}'.format(results['test_mcc'].mean()))
    print('mcc, std: {:0.3}'.format(results['test_mcc'].std()))
    print(" - - - - - - - - - - - - - - - - - - - -")

def write_performance_results_to_file(results, model, model_name, output_file_path="scores.csv"):
    with open(output_file_path, "w") as ff:
        ff.write("\nmethod,value\n")
        ff.write('f1_score_mean,{:0.3}\n'.format(results['test_f1_score'].mean()))
        ff.write('f1_score_std,{:0.3}\n'.format(results['test_f1_score'].std()))
        ff.write('roc_auc_score_mean,{:0.3}\n'.format(results['test_roc_auc_score'].mean()))
        ff.write('roc_auc_score_std,{:0.3}\n'.format(results['test_roc_auc_score'].std()))
        ff.write('accuracy_mean,{:0.3}\n'.format(results['test_accuracy'].mean()))
        ff.write('accuracy_std,{:0.3}\n'.format(results['test_accuracy'].std()))
        ff.write('precision_mean,{:0.3}\n'.format(results['test_precision'].mean()))
        ff.write('precision_std,{:0.3}\n'.format(results['test_precision'].std()))
        ff.write('recall_mean,{:0.3}\n'.format(results['test_recall'].mean()))
        ff.write('recall_std,{:0.3}\n'.format(results['test_recall'].std()))
        ff.write('specificity_mean,{:0.3}\n'.format(results['test_specificity'].mean()))
        ff.write('specificity_std,{:0.3}\n'.format(results['test_specificity'].std()))
        ff.write('mcc_mean,{:0.3}\n'.format(results['test_mcc'].mean()))
        ff.write('mcc_std,{:0.3}\n'.format(results['test_mcc'].std()))

    with open(output_file_path.replace(".csv", "-BestParams.txt"), "w") as ff:
        import json
        ff.write(json.dumps(model.named_steps[model_name].best_params_, indent=3))

def print_performance_results_for_test_data(model, X_test, y_test, columns_to_drop = []):
    predictions = model.predict(X_test)
    print("\n - - Test performance results - -\n")
    print("accuracy_score: {:0.3}".format(accuracy_score(y_test, predictions)))
    print("precision_score: {:0.3}".format(precision_score(y_test, predictions)))
    print("recall_score: {:0.3}".format(recall_score(y_test, predictions)))
    print("f1_score: {:0.3}".format(f1_score(y_test, predictions)))
    print("roc_auc_score: {:0.3}".format(roc_auc_score(y_test, predictions)))
    print("specificity_score: {:0.3}".format(specificity_score(y_test, predictions)))
    print("matthews_corrcoef: {:0.3}".format(matthews_corrcoef(y_test, predictions, sample_weight=None)))

def write_performance_results_to_file_for_test_data(model, X_test, y_test, columns_to_drop = [], output_file_path="TestDataScores.csv"):
        predictions = model.predict(X_test)
        with open(output_file_path, "w") as ff:
            ff.write("method,value\n")
            ff.write("accuracy_score,{:0.3}\n".format(accuracy_score(y_test, predictions)))
            ff.write("precision_score,{:0.3}\n".format(precision_score(y_test, predictions)))
            ff.write("recall_score,{:0.3}\n".format(recall_score(y_test, predictions)))
            ff.write("f1_score,{:0.3}\n".format(f1_score(y_test, predictions)))
            ff.write("roc_auc_score,{:0.3}\n".format(roc_auc_score(y_test, predictions)))
            ff.write("specificity_score,{:0.3}\n".format(specificity_score(y_test, predictions)))
            ff.write("matthews_corrcoef,{:0.3}\n".format(matthews_corrcoef(y_test, predictions, sample_weight=None)))



#######################################################################
## Creating models
#######################################################################
scoring = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score),
           'recall': make_scorer(recall_score),
           'f1_score': make_scorer(f1_score),
           'roc_auc_score': make_scorer(roc_auc_score),
           'mcc': make_scorer(matthews_corrcoef),
           'specificity': make_scorer(specificity_score)}


###############
### GaussianNB
###############
def run_GNB(X_train,
            X_test,
            y_train,
            y_test,
            input_file_path,
            output_root_dir,
            colsdel,
            n_splits = 10,
            n_repeats = 10,
            n_jobs = 8,
            postfix = "",
            save_model = 0,
            use_stan_scaler = 0):
    print("\n Working on the GaussianNB model . . .")

    estimators = []
    if use_stan_scaler > 0:
        estimators.append(('std', StandardScaler()))
    estimators.append(('GNB', GridSearchCV(GaussianNB(),
                                            hparams.GNB_parameter_space,
                                            cv = RepeatedStratifiedKFold(n_splits=n_splits , n_repeats= n_repeats),
                                            n_jobs= n_jobs, refit= 'roc_auc_score', scoring = scoring)))
    model = Pipeline(estimators)
    with parallel_backend('threading', n_jobs = n_jobs):
        results = cross_validate(model,  X_train, y_train, n_jobs = n_jobs, cv = RepeatedStratifiedKFold(n_splits=n_splits , n_repeats= n_repeats), scoring = scoring, return_estimator=True)

    with parallel_backend('threading', n_jobs = n_jobs):
        model.fit(X_train,y_train)

    if save_model > 0:
        tools.save_model_joblib(model, "", "models/GNB-{}.joblib".format(postfix))
        # You can load models with load_model_joblib method

    model_name = "GNB"
    print_performance_results(results, model, model_name)
    write_performance_results_to_file(results, model, model_name, "{}GNB-{}-trainData.csv".format(output_root_dir, postfix))
    print_performance_results_for_test_data(model, X_test, y_test, colsdel,)
    write_performance_results_to_file_for_test_data(model, X_test, y_test, colsdel, "{}GNB-{}-testData.csv".format(output_root_dir, postfix))



###############
### KNeighborsClassifier
###############
def run_KNN(X_train,
            X_test,
            y_train,
            y_test,
            input_file_path,
            output_root_dir,
            colsdel,
            n_splits = 10,
            n_repeats = 10,
            n_jobs = 8,
            postfix = "",
            save_model = 0,
            use_stan_scaler = 0):
    print("\n Working on the KNeighborsClassifier model . . .")

    estimators = []
    if use_stan_scaler > 0:
        estimators.append(('std', StandardScaler()))
    estimators.append(('KNN', GridSearchCV(KNeighborsClassifier(),
                                            hparams.KNN_parameter_space,
                                            cv = RepeatedStratifiedKFold(n_splits=n_splits , n_repeats=n_repeats),
                                            n_jobs=n_jobs, refit= 'roc_auc_score', scoring = scoring )))
    model = Pipeline(estimators)
    with parallel_backend('threading', n_jobs = n_jobs):
        results = cross_validate(model,  X_train, y_train, n_jobs = n_jobs, cv = RepeatedStratifiedKFold(n_splits=n_splits , n_repeats= n_repeats), scoring = scoring, return_estimator=True)

    with parallel_backend('threading', n_jobs = n_jobs):
        model.fit(X_train,y_train)

    if save_model > 0:
        tools.save_model_joblib(model, "", "models/KNN-{}.joblib".format(postfix))
        # You can load models with load_model_joblib method

    model_name = "KNN"
    print_performance_results(results, model, model_name)
    write_performance_results_to_file(results, model, model_name, "{}KNN-{}-trainData.csv".format(output_root_dir, postfix))
    print_performance_results_for_test_data(model, X_test, y_test, colsdel,)
    write_performance_results_to_file_for_test_data(model, X_test, y_test, colsdel, "{}KNN-{}-testData.csv".format(output_root_dir, postfix))


###################
####    SVM
###################
def run_SVM(X_train,
            X_test,
            y_train,
            y_test,
            input_file_path,
            output_root_dir,
            colsdel,
            n_splits = 10,
            n_repeats = 10,
            n_jobs = 8,
            postfix = "",
            save_model = 0,
            use_stan_scaler = 0):
    print("\n Working on the SVM model . . .")

    estimators = []
    if use_stan_scaler > 0:
        estimators.append(('std', StandardScaler()))
    estimators.append(('SVM', GridSearchCV(SVC(probability=True),
                                            hparams.SVM_parameter_space,
                                            cv = RepeatedStratifiedKFold(n_splits=n_splits , n_repeats=n_repeats),
                                            n_jobs= n_jobs, refit= 'roc_auc_score', scoring = scoring )))
    model = Pipeline(estimators)
    with parallel_backend('threading', n_jobs = n_jobs):
        results = cross_validate(model,  X_train, y_train, n_jobs = n_jobs, cv = RepeatedStratifiedKFold(n_splits=n_splits , n_repeats=n_repeats), scoring = scoring)

    with parallel_backend('threading', n_jobs = n_jobs):
        model.fit(X_train,y_train)

    if save_model > 0:
        tools.save_model_joblib(model, "", "models/SVM-{}.joblib".format(postfix))
        # You can load models with load_model_joblib method

    model_name = "SVM"
    print_performance_results(results, model, model_name)
    write_performance_results_to_file(results, model, model_name, "{}SVM-{}-trainData.csv".format(output_root_dir, postfix))
    print_performance_results_for_test_data(model,X_test, y_test, colsdel,)
    write_performance_results_to_file_for_test_data(model,X_test, y_test, colsdel, "{}SVM-{}-testData.csv".format(output_root_dir, postfix))


##################
###    RF
##################
def run_RF(X_train,
            X_test,
            y_train,
            y_test,
            input_file_path,
            output_root_dir,
            colsdel,
            n_splits = 10,
            n_repeats = 10,
            n_jobs = 8,
            postfix = "",
            save_model = 0,
            use_stan_scaler = 0):
    print("\n Working on the RF model . . .")

    estimators = []
    if use_stan_scaler > 0:
        estimators.append(('std', StandardScaler()))
    estimators.append(('RandomForest', GridSearchCV(RandomForestClassifier(),
                                                    hparams.RF_parameter_space,
                                                    cv = RepeatedStratifiedKFold(n_splits=n_splits , n_repeats=n_repeats),
                                                    n_jobs= n_jobs, refit= 'roc_auc_score', scoring = scoring )))
    model = Pipeline(estimators)

    with parallel_backend('threading', n_jobs = n_jobs):
        results = cross_validate(model,  X_train, y_train, n_jobs = n_jobs, cv = RepeatedStratifiedKFold(n_splits=n_splits , n_repeats=n_repeats), scoring = scoring, return_estimator=True)

    with parallel_backend('threading', n_jobs = n_jobs):
        model.fit(X_train,y_train)

    if save_model > 0:
        tools.save_model_joblib(model, "", "models/RF-{}.joblib".format(postfix))
        # You can load models with load_model_joblib method

    model_name = "RandomForest"
    print_performance_results(results, model, model_name)
    write_performance_results_to_file(results, model, model_name, "{}RF-{}-trainData.csv".format(output_root_dir, postfix))
    print_performance_results_for_test_data(model, X_test, y_test, colsdel,)
    write_performance_results_to_file_for_test_data(model, X_test, y_test, colsdel, "{}RF-{}-testData.csv".format(output_root_dir, postfix))


###################
####    XGBoost
###################
def run_XGB(X_train,
            X_test,
            y_train,
            y_test,
            input_file_path,
            output_root_dir,
            colsdel,
            n_splits = 10,
            n_repeats = 10,
            n_jobs = 8,
            postfix = "",
            save_model = 0,
            use_stan_scaler = 0):
    print("\n Working on the XGBoost model . . .")

    estimators = []
    if use_stan_scaler > 0:
        estimators.append(('std', StandardScaler()))
    estimators.append(('XGBClassifier', GridSearchCV(XGBClassifier(use_label_encoder=False),
                                                                hparams.XGB_parameter_space,
                                                                cv = RepeatedStratifiedKFold(n_splits=n_splits , n_repeats=n_repeats),
                                                                n_jobs=n_jobs, refit= 'roc_auc_score', scoring = scoring)))
    model = Pipeline(estimators)
    with parallel_backend('threading', n_jobs = n_jobs):
        results = cross_validate(model,  X_train, y_train, n_jobs = n_jobs, cv = RepeatedStratifiedKFold(n_splits=n_splits , n_repeats=n_repeats), scoring = scoring)

    with parallel_backend('threading', n_jobs = n_jobs):
        model.fit(X_train,y_train)

    if save_model > 0:
        tools.save_model_joblib(model, "", "models/XGBoost-{}.joblib".format(postfix))
        # You can load models with load_model_joblib method

    model_name = "XGBClassifier"
    print_performance_results(results, model, model_name)
    write_performance_results_to_file(results, model, model_name, "{}XGBoost-{}-trainData.csv".format(output_root_dir, postfix))
    print_performance_results_for_test_data(model, X_test, y_test, colsdel,)
    write_performance_results_to_file_for_test_data(model, X_test, y_test, colsdel, "{}XGBoost-{}-testData.csv".format(output_root_dir, postfix))


#######################################################################
#######################################################################
#######################################################################
if	__name__ == "__main__":
    usage ='''
    USAGE:
    python main.py -i input_file_path
                   -l label_col_name
                   -o output_root_dir
                   [-c col_name_to_delete]
                   [-s cv_splits]
                   [-r n_cv_repeats]
                   [-j n_jobs]
                   [-p output_name_postfix]
                   [-m model_names]
                   [-s 0 or 1]

    -h  (--help)        Shows these instructions :)
    -i  (--input)       The input file path
    -l  (--labelcol)    The labels' column name
    -o  (--out)         (default="") The output root directory
    -c  (--colsdel)     (default="") Column names you want to be removed from the
                        training data. e.g. sample_name1,sample_name2
    -k  (--kfold)       (default=10) The value of  k  for k-fold cross validation.
                        i.e. The number of folds
    -r  (--repeats)     (default=10) The number of iterations for the cross validation stage
    -j  (--jobs)        (default=8) Number of threads for running modles
    -p  (--postfix)     (default="") The postfix for output file names
    -m  (--models)      (defaults=rf,svm,xgb) Models that you want to build and test.
                        divide names with ','. e.g.  rf,svm,xgb,knn,gnb
                        Available machines:
                            RF  = Random Forest
                            SVM = Support Vector Machine
                            XGB = eXtreme Gradient Boosting
                            KNN = K-Nearest Neighbors
                            GNB = Gussian Naive Bayes
    -s  (--save)        (default=0) Set it to 1 if you want to save your model as a
                        joblib file.
    -t  (--testsplit)   (default=0.25) The test split size
    -x  (--stanscale)   (default=0) Set it to 1 if you want to standardize features
    '''

    input_file_path = ""
    labelcol = ""
    output_root_dir = ""
    n_splits = 10
    n_repeats = 10
    n_jobs = 8
    postfix = ""
    models = ["rf", "svm", "xgb"]
    colsdel = []
    save_model = 0
    test_split_size = 0.25
    use_stan_scaler = 0


    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:k:r:j:p:o:m:l:c:s:t:x", ["input=", "kfold=", "repeats=", "jobs=", "postfix=", "out=", "models=", "labelcol=", "colsdel=", "save=", "testsplit=", "stanscale="])
    except getopt.GetoptError:
        print(usage)
        sys.exit()
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(usage)
            sys.exit()
        if opt in ("-i", "--input"):
            try:
                if len(arg) > 0 and os.path.isfile(arg):
                    input_file_path = arg.replace("\\", "/").replace("\"", "").replace("'", "")
                else:
                    raise ValueError("Input file path not found: \n  >>>    {}\n".format(arg))
            except Exception as e:
                print("ERROR: {} \n {}".format(e, usage))
                sys.exit()
        if opt in ("-l", "--labelcol"):
            try:
                if len(arg) > 0:
                    labelcol = arg
                else:
                    raise ValueError("labels' column name is empty\n")
            except Exception as e:
                print("ERROR: {} \n {}".format(e, usage))
                sys.exit()
        if opt in ("-o", "--out"):
            try:
                if len(arg) > 0:
                    output_root_dir = arg.replace("\\", "/").replace("\"", "").replace("'", "")
                    if output_root_dir[-1] != "/":
                        output_root_dir += "/"
                else:
                    raise ValueError("Please enter the output root directory")
            except Exception as e:
                print("ERROR: {} \n {}".format(e, usage))
                sys.exit()
        if opt in ("-c", "--colsdel"):
            try:
                colsdel.extend(arg.lower().split(","))
            except Exception as e:
                print(usage)
                sys.exit()
        if opt in ("-k", "--kfold"):
            try:
                n_splits = int(arg)
            except Exception as e:
                print("ERROR: input parameters for 'splits' should be INTEGER \n {}".format(usage))
                sys.exit()
        if opt in ("-r", "--repeats"):
            try:
                n_repeats = int(arg)
            except Exception as e:
                print("ERROR: input parameters for 'repeats' should be INTEGER \n {}".format(usage))
                sys.exit()
        if opt in ("-j", "--jobs"):
            try:
                n_jobs = int(arg)
            except Exception as e:
                print("ERROR: input parameters for 'jobs' should be INTEGER \n {}".format(usage))
                sys.exit()
        if opt in ("-p", "--postfix"):
            try:
                postfix = (arg)
            except Exception as e:
                print(usage)
                sys.exit()
        if opt in ("-m", "--models"):
            try:
                if len(arg) > 0:
                    models = []
                    models.extend(arg.lower().split(","))
            except Exception as e:
                print(usage)
                sys.exit()
        if opt in ("-s", "--save"):
            try:
                if int(arg) > 0:
                    save_model = 1
                else:
                    save_model = 0
            except Exception as e:
                print("ERROR: input parameters for 'save' should be INTEGER \n {}".format(usage))
                sys.exit()
        if opt in ("-t", "--testsplit"):
            try:
                if float(arg) >=0 and float(arg) <= 1:
                    test_split_size = float(arg)
                else:
                    raise ValueError("")
            except Exception as e:
                print("ERROR: input parameters for 'testsplit' should be a number between 0 and 1 \n {}".format(usage))
                sys.exit()
        if opt in ("-x", "--stanscale"):
            try:
                if int(arg) > 0:
                    use_stan_scaler = 1
                else:
                    use_stan_scaler = 0
            except Exception as e:
                print("ERROR: input parameters for 'stanscale' should be INTEGER \n {}".format(usage))
                sys.exit()


    # Check for mandatory inputs
    if len(input_file_path) == 0 or len(labelcol) == 0 or len(output_root_dir) == 0:
        print(usage)
        sys.exit()

    colsdel.append(labelcol)


    input_data = pd.read_csv(input_file_path)
    X = input_data.drop(colsdel, axis=1)
    y = input_data[labelcol]
    X_train , X_test , y_train , y_test = train_test_split (X, y, test_size=test_split_size, random_state = 42)
    print("Train shape: {}  | Test shape: {}".format(y_train.shape, X_train.shape))

    for model in models:
        if model == "xgb":
            run_XGB(X_train, X_test, y_train, y_test,
                    input_file_path, output_root_dir, colsdel,
                    n_splits, n_repeats, n_jobs, postfix,
                    save_model, use_stan_scaler)
        if model == "rf":
            run_RF(X_train, X_test, y_train, y_test,
                    input_file_path, output_root_dir, colsdel,
                    n_splits, n_repeats, n_jobs, postfix,
                    save_model, use_stan_scaler)
        if model == "svm":
            run_SVM(X_train, X_test, y_train, y_test,
                    input_file_path, output_root_dir, colsdel,
                    n_splits, n_repeats, n_jobs, postfix,
                    save_model, use_stan_scaler)
        if model == "knn":
            run_KNN(X_train, X_test, y_train, y_test,
                    input_file_path, output_root_dir, colsdel,
                    n_splits, n_repeats, n_jobs, postfix,
                    save_model, use_stan_scaler)
        if model == "gnb":
            run_GNB(X_train, X_test, y_train, y_test,
                    input_file_path, output_root_dir, colsdel,
                    n_splits, n_repeats, n_jobs, postfix,
                    save_model, use_stan_scaler)
