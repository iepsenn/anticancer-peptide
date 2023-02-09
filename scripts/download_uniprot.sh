#!/usr/bin/env bash

mkdir -p data/raw/uniprot/
wget -O data/raw/uniprot/swissprot.fasta.gz 'https://rest.uniprot.org/uniprotkb/stream?compressed=true&download=true&format=fasta&query=reviewed%3Atrue'
cd data/raw/uniprot/
gzip -d -f data/raw/uniprot/swissprot.fasta.gz