import sys
import os
import glob
import numpy as np
import pandas as pd

from Bio import SeqIO, SearchIO
from Bio.Blast.Applications import NcbimakeblastdbCommandline, NcbiblastnCommandline


database = sys.argv[1]
gtf = sys.argv[2]
fasta_file = sys.argv[3]


gtf_table = pd.read_csv(gtf, sep='\t', header=None)
gtf_table = gtf_table.loc[gtf_table[2] == 'CDS', :]

db_files = [database + ending for ending in ['.nsq', '.nhr', '.nin']]

# Build blast database from genome if it doesn't already exist
if not all([os.path.exists(f) for f in db_files]):
	cline = NcbimakeblastdbCommandline(dbtype='nucl', input_file=database)
	sys.stderr.write('No blast database, building it' + os.linesep)
	sys.stderr.write(str(cline) + os.linesep)
	cline()

# Get all hits
coords = []
genes = []
with open(fasta_file, 'r') as handle:
	for record in SeqIO.parse(handle, 'fasta'):
		with open(record.id + '.fasta', 'w') as outfa:
			SeqIO.write(record, outfa, 'fasta')
		blastn = NcbiblastnCommandline(query=record.id + '.fasta', db=database, out=record.id + '.xml', evalue=1000, word_size=13, gapopen=5, gapextend=2, outfmt=5)
		blastn()
		os.remove(record.id + '.fasta')
		results = SearchIO.read(record.id + '.xml', 'blast-xml')
		
		if len(results) > 0:
			for top_hit in results:
			  for hsp in top_hit:
				  coords.append(hsp.hit_range)
				  genes.append(hsp.hit_id)
			os.remove(record.id + '.xml')
		else:
			  print(f'No results for {record.id}')
			  exit()

# Save Results in useful format
gtf_search = []

for gene in genes:
  gtfhit_table = gtf_table.loc[gtf_table[0] == gene, :]
  print(f'Match for gene')
  gtf_search.append(gtfhit_table[[0, 3, 4, 8]])
gtf_entries = pd.concat(gtf_search)
print(gtf_entries.head())
gtf_entries.to_csv(fasta_file.replace('fasta', 'bed'), header=False, index=False, sep='\t')

