import sys
import os
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, make_scorer, cohen_kappa_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from collections import Counter
from pathlib import Path
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate, cross_val_predict

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category= FutureWarning)
simplefilter(action='ignore', category= UserWarning)
simplefilter(action='ignore', category= DeprecationWarning)


def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]
scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tn),
           'fp': make_scorer(fp), 'fn': make_scorer(fn),
           'auc': 'roc_auc',
           'acc': make_scorer(accuracy_score),
           'kappa': make_scorer(cohen_kappa_score),
           'prec': make_scorer(precision_score),
           'rec': make_scorer(recall_score)}



def update_progress(progress):
    barLength = 100 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), round(progress*100, 2), status)
    sys.stdout.write(text)
    sys.stdout.flush()  

if __name__ == "__main__":
    method = "10000Best"
    
    # Nested Cross Validation:
    outer_loop_cv = 5
    
    # Number of random trials:
    NUM_TRIALS = 50

    # Initialize Variables:
    scores_auc = np.zeros(NUM_TRIALS)
    scores_acc = np.zeros(NUM_TRIALS)
    scores_sens = np.zeros(NUM_TRIALS)
    scores_spec = np.zeros(NUM_TRIALS)
    scores_kappa = np.zeros(NUM_TRIALS)
    scores_prec = np.zeros(NUM_TRIALS)
    scores_rec = np.zeros(NUM_TRIALS)

    name_antibiotic_list = [d for d in os.listdir() if os.path.isdir(os.path.join(d))]

    for name_antibiotic in name_antibiotic_list:
        if name_antibiotic == "results":
            continue
        print("Antibiotic: {}".format(name_antibiotic))

        file_name = name_antibiotic+"/SMOTE_"+method+"_results_"+name_antibiotic+'.csv'
        my_file = Path(file_name)

        try:
            my_abs_path = my_file.resolve(strict=True)
        except FileNotFoundError:
            pass
        else:
            continue

        # Load Data:
        antibiotic_df = pd.read_csv(name_antibiotic+"/"+name_antibiotic+'_AMR.csv', header = [0])

        n_lines = antibiotic_df.shape[0]   
        print("Number of isolates = {}".format(n_lines)) 
        
        target_str = np.array(antibiotic_df[antibiotic_df.columns[1]])
        
        target = np.zeros(len(target_str)).astype(int)
        idx_S = np.where(target_str == 'Susceptible')[0]
        idx_R = np.where(target_str == 'Resistant')[0]
        idx_NaN = np.where((target_str != 'Resistant') & (target_str != 'Susceptible'))[0]
        target[idx_R] = 1    

        if len(idx_NaN) > 0:
            target = np.delete(target,idx_NaN)
            n_lines = len(target)
            print("Correct number of isolates: {}".format(len(target)))


        count_class = Counter(target)
        print(count_class)

        if count_class[0] < 12 or count_class[1] < 12:
            continue

        file_name = name_antibiotic+"/data_"+method+"_"+name_antibiotic+'.pickle'
        my_file = Path(file_name)

        try:
            my_abs_path = my_file.resolve(strict=True)
        except FileNotFoundError:
            continue
        else:
            with open(file_name, 'rb') as f:
                data = pickle.load(f)

        # Standardize data: zero mean and unit variance
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

        # Get k-mers
        features_df = pd.read_csv(name_antibiotic+"/"+name_antibiotic+"_"+method+"_pvalue.csv", header = None)
        n_features = features_df.shape[0]

        print("Number of features = {}".format(n_features))

        kmers_list = np.array(features_df.loc[:,features_df.columns[0]]).astype(int)
    

        summary = pd.DataFrame(index=kmers_list, columns=['Mean importance', 'Max importance', 'Frequency'], data=np.zeros((len(kmers_list), 3), dtype=np.int32))
        # Loop for each trial
        update_progress(0)
        for i in range(NUM_TRIALS):
            #print("Trial = {}".format(i))
        
            outer_cv = StratifiedKFold(n_splits=outer_loop_cv, shuffle=True, random_state=i)
        
            model = Pipeline([
                ('sampling',SMOTE(random_state=i)),
                ('clf', GradientBoostingClassifier())
            ])

            # Outer Search
            cv_results = cross_validate(model, data, target, scoring=scoring, cv=outer_cv)
                
            tp = cv_results['test_tp']
            tn = cv_results['test_tn']
            fp = cv_results['test_fp']
            fn = cv_results['test_fn']
            
            sens = np.zeros(outer_loop_cv)
            spec = np.zeros(outer_loop_cv)

            for j in range(outer_loop_cv):
                TP = tp[j]
                TN = tn[j]
                FP = fp[j]
                FN = fn[j]
                
                # Sensitivity, hit rate, recall, or true positive rate
                sens[j] = TP/(TP+FN)
                
                # Fall out or false positive rate
                FPR = FP/(FP+TN)
                spec[j] = 1 - FPR

            scores_sens[i] = sens.mean()
            scores_spec[i] = spec.mean()
            scores_auc[i] = cv_results['test_auc'].mean()
            scores_acc[i] = cv_results['test_acc'].mean()
            scores_kappa[i] = cv_results['test_kappa'].mean()
            scores_prec[i] = cv_results['test_prec'].mean()
            scores_rec[i] = cv_results['test_rec'].mean()

            for train_index, test_index in outer_cv.split(data, target):
                model.fit(data[train_index,:], target[train_index])
                ypred = model.predict(data[test_index,:])       
                ytest = target[test_index]  

                top_feats = np.argwhere(model.named_steps["clf"].feature_importances_ > 0).reshape(-1)
                #print(top_feats)

                kmers_top = kmers_list[top_feats]
                summary.loc[kmers_top, 'Mean importance'] += model.named_steps["clf"].feature_importances_[model.named_steps["clf"].feature_importances_ > 0].reshape(-1)
                for kmer, x in zip(kmers_top, model.named_steps["clf"].feature_importances_[model.named_steps["clf"].feature_importances_ > 0].reshape(-1)):
                    if x > summary.loc[kmer, 'Max importance']:
                        summary.loc[kmer, 'Max importance'] = x
                summary.loc[kmers_top, 'Frequency'] +=1
            
            update_progress((i+1)/NUM_TRIALS)

        summary.loc[:, 'Mean importance'] /= outer_loop_cv*NUM_TRIALS
        summary.loc[summary['Frequency'] > 2.5].to_csv(name_antibiotic+"/SMOTE_"+method+"_"+name_antibiotic+'_average_kmers.csv', sep=',')
        
        results = np.zeros((7,2))
        scores = [scores_auc, scores_acc, scores_sens, scores_spec, scores_kappa, scores_prec, scores_rec]
        for counter_scr, scr in enumerate(scores):
            results[counter_scr,0] = np.mean(scr,axis=0)
            results[counter_scr,1] = np.std(scr,axis=0)
            
        names_scr = ["AUC", "Acc", "Sens", "Spec", "Kappa", "Prec", "Recall"]

        results_df=pd.DataFrame(results, columns=["Mean", "Std"], index=names_scr)

        np.savetxt(name_antibiotic+"/SMOTE_"+method+"_"+name_antibiotic+"_auc.csv", scores_auc, delimiter=",")
        np.savetxt(name_antibiotic+"/SMOTE_"+method+"_"+name_antibiotic+"_acc.csv", scores_acc, delimiter=",")
        np.savetxt(name_antibiotic+"/SMOTE_"+method+"_"+name_antibiotic+"_sens.csv", scores_sens, delimiter=",")
        np.savetxt(name_antibiotic+"/SMOTE_"+method+"_"+name_antibiotic+"_spec.csv", scores_spec, delimiter=",")
        np.savetxt(name_antibiotic+"/SMOTE_"+method+"_"+name_antibiotic+"_kappa.csv", scores_kappa, delimiter=",")
        np.savetxt(name_antibiotic+"/SMOTE_"+method+"_"+name_antibiotic+"_prec.csv", scores_prec, delimiter=",")
        np.savetxt(name_antibiotic+"/SMOTE_"+method+"_"+name_antibiotic+"_rec.csv", scores_rec, delimiter=",")
        results_df.to_csv(name_antibiotic+"/SMOTE_"+method+"_results_"+name_antibiotic+".csv")


