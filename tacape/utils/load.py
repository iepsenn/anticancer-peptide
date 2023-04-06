from Bio import SeqIO

def load_data(file, format):
    if format == 'text':
        return [line.upper().strip('\n') for line in open(file)]
    elif format == 'fasta':
        return [str(record.seq).upper() for record in SeqIO.parse(file, 'fasta')]
    else:
        raise Exception(f'format "{format}" not supported')