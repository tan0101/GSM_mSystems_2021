import sys
from Bio import SeqIO, SearchIO
from Bio.Blast.Applications import NcbimakeblastdbCommandline, NcbiblastnCommandline

fasta_file = sys.stdin

with fasta_file as handle:
	for record in SeqIO.parse(handle, 'fasta'):
		with open(record.id + '.fasta', 'w') as outfa:
			SeqIO.write(record, outfa, 'fasta')
