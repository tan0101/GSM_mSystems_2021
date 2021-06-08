# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
import math
import pickle

from sklearn.feature_selection import chi2, SelectKBest

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
    
    name_antibiotic = "ampicillin" 
    method = "10000Best"
    n_Best = 10000

    print(name_antibiotic)
    
    # Load Data:
    antibiotic_df = pd.read_csv(name_antibiotic+"/"+name_antibiotic+'_AMR.csv', header = [0])

    n_lines = antibiotic_df.shape[0]   
    print("Number of isolates = {}".format(n_lines)) 
    delimiter = ' '

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

    
    for n, line in enumerate(open(name_antibiotic+"/"+name_antibiotic+'_kmer_data.txt','r')):
        if n == 0:
            dummy = np.array(line.split(delimiter), dtype=float)
            n_columns = dummy.shape[0]
            break

    print("Number of kmers = {}".format(n_columns))

    steps = 2000000
    floor_value = math.floor(n_columns/steps)
    ceil_value = math.ceil(n_columns/steps)
    print("floor value = {}".format(floor_value))
    cols_all = np.zeros(n_Best*math.ceil(n_columns/steps))
    k = 0
    ni = 0
    update_progress(0)
    for i in range(n_columns):
        ni += 1
        if k < floor_value:
            if ni == steps:
                subset_data = np.zeros((n_lines,steps),dtype=int)
                ind_array = np.arange(steps*k,steps*(k+1))
                update_progress(0)
                num_line = 0
                with open(name_antibiotic+"/"+name_antibiotic+'_kmer_data.txt','r') as file:
                    for n, line in enumerate(file):
                        if n in idx_NaN:
                            continue
                        else:
                            dummy = np.array(line.split(delimiter)) 
                            subset_data[num_line,:] = dummy[ind_array].astype(int)
                            num_line += 1

                        update_progress((num_line+1)/n_lines)

                selector = SelectKBest(chi2,k=n_Best)
                dummy_sel = selector.fit_transform(subset_data, target)
                sup_ind = selector.get_support(indices=True)
                cols_all[n_Best*k:n_Best*(k+1)] = ind_array[sup_ind]
                del subset_data
                print("k = {}".format(k))
                k += 1
                ni = 0
        else:
            if i == n_columns - 1:
                subset_data = np.zeros((n_lines,ni),dtype=int)
                ind_array = np.arange(steps*k,n_columns)
                update_progress(0)
                num_line = 0
                with open(name_antibiotic+"/"+name_antibiotic+'_kmer_data.txt','r') as file:
                    for n, line in enumerate(file):
                        if n in idx_NaN:
                            continue
                        else:
                            line = line.strip('\n')
                            dummy = np.array(line.split(delimiter))                  
                            subset_data[num_line,:] = dummy[ind_array].astype(int)
                            num_line += 1
                        update_progress((num_line+1)/n_lines)

                selector = SelectKBest(chi2,k=n_Best)
                dummy_sel = selector.fit_transform(subset_data, target)
                sup_ind = selector.get_support(indices=True)
                cols_all[n_Best*k:] = ind_array[sup_ind]
                del subset_data
        if i % 10000 == 0:
            update_progress((i+1)/n_columns)

    print("Finshed - Find 2000 best now")
    
    update_progress(0)
    cols_all = cols_all.astype(int)
    np.savetxt(name_antibiotic+"/cols_all_"+method+'_'+name_antibiotic+'.txt', cols_all, fmt='%d')        
    
    data = np.zeros((n_lines,n_Best*ceil_value))
    num_line = 0
    with open(name_antibiotic+"/"+name_antibiotic+'_kmer_data.txt','r') as file:
        for n, line in enumerate(file):
            if n in idx_NaN:
                continue
            else:
                dummy = np.array(line.split(delimiter), dtype=float)
                data[num_line,:] = dummy[cols_all]
                num_line += 1
            update_progress((n+1)/n_lines)

    selector = SelectKBest(chi2,k=n_Best)
    data = selector.fit_transform(data, target)
    print(data.shape)
    cols_dummy = selector.get_support(indices=True)
    cols = cols_all[cols_dummy]
    scores = selector.scores_
    print(len(scores))
    pvalues = selector.pvalues_
    print(len(pvalues))

    concat_array = np.zeros((len(cols),3))
    concat_array[:,0] = cols
    concat_array[:,1] = scores[cols_dummy]
    concat_array[:,2] = pvalues[cols_dummy]

    np.savetxt(name_antibiotic+"/features_"+method+'_'+name_antibiotic+'.txt', cols, fmt='%d')        
    np.savetxt(name_antibiotic+"/"+name_antibiotic+"_"+method+"_pvalue.csv", concat_array, delimiter=",")

    with open(name_antibiotic+"/data_"+method+'_'+name_antibiotic+'.pickle', 'wb') as f:
        pickle.dump(data, f)
