#!/bin/bash


if [[ $1 == "--from-scratch" ]] 
then
	echo "Deleting all solution files"
	rm -rf solutions/*
fi

cd knn_app
make > /dev/null


delta=0.001
sample=0.5
n_trees=20
threads=8

for dir in ../datasets/sigmod; 
do
    k=$([ "sigmod" == $(basename $dir) ] && echo "100" || echo "20")
    dim=$([ "sigmod" == $(basename $dir) ] && echo "100" || echo "4")
    endian=$([ "sigmod" == $(basename $dir) ] && echo "little" || echo "big")

    for file in "$dir"/*;
    do
        for metric in euclidean_opt
        do
		    echo Testing: $file with K = $k and $metric metric, delta = $delta, sample rate = $sample, threads = $threads, n_trees = $n_trees
		    ../build/knn_app -file $file -k $k -endian $endian -dim $dim -metric $metric -precision $delta -sample $sample -threads $threads -trees $n_trees
		    echo "-------------------------------------------------------------------------------------------------------------"
	    done                 
    done
done

make clean > /dev/null
