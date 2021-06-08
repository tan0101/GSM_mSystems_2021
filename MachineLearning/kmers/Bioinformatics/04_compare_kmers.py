import sys
import os
import glob
import numpy as np
import pandas as pd

from subprocess import Popen, PIPE
#from scipy.stats import chisquare
from sklearn.feature_selection import chi2

def get_kmer_count_dict(files, kmer_list):
	n_files = len(files)
	n_kmers = len(kmer_list)
	data = pd.DataFrame(index=files)
	for f in files:
		with Popen(['glistquery', f, '-s', kmer_list], stdout=PIPE) as p:
			for line in p.stdout.readlines():
				kmer, num = line.decode('utf-8').strip().split('\t')
				data.loc[f, kmer] = int(num)
	#for key in data.keys():
	#	data[key] /= n_files
	return data

sus = sys.argv[1]
res = sys.argv[2]
kmers = sys.argv[3]

sus_files = glob.glob(os.path.join(sus, '*.list'))
res_files = glob.glob(os.path.join(res, '*.list'))

sus_data = get_kmer_count_dict(sus_files, kmers)
res_data = get_kmer_count_dict(res_files, kmers)
df = pd.concat([sus_data, res_data], axis=0, ignore_index=True)
labels = np.concatenate([np.zeros(shape=(len(sus_files), 1)), np.ones(shape=(len(res_files), 1))])
# sort out chisquare calculations
chi2_res = chi2(df, labels)
chi2_data = pd.DataFrame({'kmers': df.columns, 'chi2': chi2_res[0], 'pvalue': chi2_res[1]})
chi2_data['sum_sus'] = [sus_data[kmer].sum() for kmer in chi2_data['kmers']]
chi2_data['sum_res'] = [res_data[kmer].sum() for kmer in chi2_data['kmers']]
chi2_data.to_csv('kmer_comps.txt', sep='\t', index=False)
