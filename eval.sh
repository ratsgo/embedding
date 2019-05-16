#!/usr/bin/env bash

COMMAND=$1

case $COMMAND in
    download)
        echo "download test dataset..."
        wget https://github.com/dongjun-Lee/kor2vec/raw/master/test_dataset/kor_ws353.csv
        wget https://github.com/dongjun-Lee/kor2vec/raw/master/test_dataset/kor_analogy_semantic.txt
        wget https://github.com/dongjun-Lee/kor2vec/raw/master/test_dataset/kor_analogy_syntactic.txt
        mv kor* data
        ;;
    word2vec)
        echo "evalation, word2vec..."

        ;;
esac