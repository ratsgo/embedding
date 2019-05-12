#!/usr/bin/env bash

pip install gensim
pip install soynlp
mkdir ../data

COMMAND=$1

case $COMMAND in
    data_dump)
        echo "download naver movie corpus..."
        wget https://github.com/e9t/nsmc/raw/master/ratings.txt
        wget https://github.com/e9t/nsmc/raw/master/ratings_train.txt
        wget https://github.com/e9t/nsmc/raw/master/ratings_test.txt
        mv *.txt ../data
        echo "download ko-wikipedia..."
        wget https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2 ../data/kowiki-latest-pages-articles.xml.bz2
        mv kowiki-latest-pages-articles.xml.bz2 ../data
        echo "processing ko-wikipedia..."
        python wikidump.py ../data/kowiki-latest-pages-articles.xml.bz2 ../data/wiki_ko_raw.txt
        ;;
    space_correct)
        echo "processing corpus..."
        python soynlp_utils.py process_nvc ../data/ratings.txt ../data/processed_ratings.txt
        cat ../data/processed_ratings.txt ../data/wiki_ko_raw.txt > ../data/corpus.txt
        echo "apply space correct..."
        python soynlp_utils.py train_space ../data/corpus.txt ../data/space.model
        python soynlp_utils.py apply_space_correct ../data/corpus.txt ../data/space.model ../data/corrected_corpus.txt
        ;;
    ltokenize)
        echo "LTokenizing..."
        python soynlp_utils.py compute_word_score ../data/tmp.txt ../data/tmp.model
esac



