#!/bin/bash

help="This is a script to download the data for the project.
Available options:
\tsmall \t for small dataset (~25MB)
\tbig \t for big dataset (~250MB)
"


if [ $# -eq 0 ]
then
    option="help"
else
    option=$1
fi

if [ $option = "small" ]
then
    mkdir -p data
    wget http://files.grouplens.org/datasets/movielens/ml-latest-small.zip -O data/data_small.zip
    cd data
    unzip -o data_small.zip
    rm data_small.zip
    cd ..
elif [ $option = "big" ]
then
    mkdir -p data
    wget http://files.grouplens.org/datasets/movielens/ml-latest.zip -O data/data_big.zip
    cd data
    unzip -o data_big.zip
    rm data_big.zip
    cd ..
else
    echo -e "$help"
fi
