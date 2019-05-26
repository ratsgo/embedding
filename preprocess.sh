#!/usr/bin/env bash

COMMAND=$1

case $COMMAND in
    data_dump)
        echo "download naver movie corpus..."
        wget https://github.com/e9t/nsmc/raw/master/ratings.txt -P /notebooks/embedding/data
        wget https://github.com/e9t/nsmc/raw/master/ratings_train.txt -P /notebooks/embedding/data
        wget https://github.com/e9t/nsmc/raw/master/ratings_test.txt -P /notebooks/embedding/data
        mv rating* data
        echo "download ko-wikipedia..."
        wget https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2 -P /notebooks/embedding/data
        mv kowiki-latest-pages-articles.xml.bz2 data -P /notebooks/embedding/data
        echo "download KorSquad data..."
        wget https://korquad.github.io/dataset/KorQuAD_v1.0_train.json -P /notebooks/embedding/data
        wget https://korquad.github.io/dataset/KorQuAD_v1.0_dev.json -P /notebooks/embedding/data
        mv KorQuAD_v1* data
        echo "download similar sentence data..."
        wget https://github.com/songys/Question_pair/raw/master/kor_pair_train.csv -P /notebooks/embedding/data
        wget https://github.com/songys/Question_pair/raw/master/kor_Pair_test.csv -P /notebooks/embedding/data
        mv kor_*air* data
        ;;
    process_wiki)
        echo "processing ko-wikipedia..."
        python preprocess/dump.py wiki data/kowiki-latest-pages-articles.xml.bz2 data/wiki_ko_raw.txt
        ;;
    process_navermovie)
        echo "processing naver movie corpus..."
        python preprocess/dump.py nsmc data/ratings.txt data/processed_ratings.txt False
        python preprocess/dump.py nsmc data/ratings_train.txt data/processed_ratings_train.txt True
        python preprocess/dump.py nsmc data/ratings_test.txt data/processed_ratings_test.txt True
        ;;
    process_korsquad)
        echo "processing KorSqaud corpus..."
        python preprocess/dump.py korsquad data/KorQuAD_v1.0_train.json data/processed_korsquad_train.txt
        python preprocess/dump.py korsquad data/KorQuAD_v1.0_dev.json data/processed_korsquad_dev.txt
        cat data/processed_korsquad_train.txt data/processed_korsquad_dev.txt > data/processed_korsquad.txt
        rm data/processed_korsquad_*.txt
        ;;
    mecab_tokenize)
        echo "mecab, tokenizing..."
        python preprocess/supervised_nlputils.py mecab data/wiki_ko_raw.txt data/wiki_ko_mecab.txt
        python preprocess/supervised_nlputils.py mecab data/processed_ratings.txt data/ratings_mecab.txt
        python preprocess/supervised_nlputils.py mecab data/processed_korsquad.txt data/korsquad_mecab.txt
        cat data/wiki_ko_mecab.txt data/ratings_mecab.txt data/korsquad_mecab.txt > data/corpus_mecab.txt
        cat data/ratings_mecab.txt data/korsquad_mecab.txt > data/for-lsa-mecab.txt
        ;;
    space_correct)
        echo "train & apply space correct..."
        python preprocess/unsupervised_nlputils.py train_space data/processed_ratings.txt data/space.model
        python preprocess/unsupervised_nlputils.py apply_space_correct data/processed_ratings.txt data/space.model data/corrected_ratings_corpus.txt False
        python preprocess/unsupervised_nlputils.py apply_space_correct data/processed_ratings_train.txt data/space.model data/corrected_ratings_train.txt True
        python preprocess/unsupervised_nlputils.py apply_space_correct data/processed_ratings_test.txt data/space.model data/corrected_ratings_test.txt True
        ;;
    soy_tokenize)
        echo "soynlp, LTokenizing..."
        python preprocess/unsupervised_nlputils.py compute_soy_word_score data/corrected_ratings_corpus.txt data/soyword.model
        python preprocess/unsupervised_nlputils.py soy_tokenize data/corrected_ratings_corpus.txt data/soyword.model data/tokenized_corpus_soynlp.txt
        ;;
    komoran_tokenize)
        echo "komoran, tokenizing..."
        python preprocess/supervised_nlputils.py komoran data/corpus.txt data/tokenized_corpus_komoran.txt
        ;;
    okt_tokenize)
        echo "okt, tokenizing..."
        python preprocess/supervised_nlputils.py okt data/corpus.txt data/tokenized_corpus_okt.txt
        ;;
    hannanum_tokenize)
        echo "hannanum, tokenizing..."
        python preprocess/supervised_nlputils.py hannanum data/corpus.txt data/tokenized_corpus_hannanum.txt
        ;;
    khai_tokenize)
        echo "khai, tokenizing..."
        python preprocess/supervised_nlputils.py khai data/corpus.txt data/tokenized_corpus_khai.txt
        ;;
    sentencepiece)
        echo "processing sentencepiece..."
        spm_train --input=data/corrected_ratings_corpus.txt --model_prefix=sentpiece --vocab_size=10000
        mv sentpiece.model data
        mv sentpiece.vocab data
        python preprocess/unsupervised_nlputils.py process_sp_vocab data/sentpiece.vocab data/processd_sentpiece.vocab
        python preprocess/unsupervised_nlputils.py sentencepiece_tokenize data/processd_sentpiece.vocab data/corrected_ratings_corpus.txt data/tokenized_corpus_sentpiece.txt
        ;;
esac