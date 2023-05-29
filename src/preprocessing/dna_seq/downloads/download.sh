#!/bin/bash

ASSEM=$1
#DIR=./downloads
DIR="/home/ubuntu/codebase/tf_binding/data/${ASSEM}/dna_sequence"

# Create dir
mkdir -p $DIR

# Download DNA data
if [ $ASSEM = "hg38" ] || [ $ASSEM = "hg19" ]
then
    a1=$(seq 1 22)
fi
if [ $ASSEM = "mm10" ] || [ $ASSEM = "mm9" ]
then
    a1=$(seq 1 19)
fi

a2=("X" "Y")
total=(${a1[@]} ${a2[@]})

for DNA in ${total[@]};
do
	curl "http://hgdownload.soe.ucsc.edu/goldenPath/${ASSEM}/chromosomes/chr${DNA}.fa.gz" -o "$DIR/chr${DNA}.fa.gz"
done

# Get DNA length
python get_chr_lengths.py $ASSEM $DIR
