#!/usr/bin/env bash

COMMAND=$1
export LC_CTYPE=C.UTF-8
function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}

case $COMMAND in
    dump-raw)
        echo "download naver movie corpus..."
        wget https://github.com/e9t/nsmc/raw/master/ratings.txt -P /notebooks/embedding/data/raw
        wget https://github.com/e9t/nsmc/raw/master/ratings_train.txt -P /notebooks/embedding/data/raw
        wget https://github.com/e9t/nsmc/raw/master/ratings_test.txt -P /notebooks/embedding/data/raw
        echo "download ko-wikipedia..."
        wget https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2 -P /notebooks/embedding/data/raw
        echo "download KorSquad data..."
        wget https://korquad.github.io/dataset/KorQuAD_v1.0_train.json -P /notebooks/embedding/data/raw
        wget https://korquad.github.io/dataset/KorQuAD_v1.0_dev.json -P /notebooks/embedding/data/raw
        echo "download similar sentence data..."
        wget https://github.com/songys/Question_pair/raw/master/kor_pair_train.csv -P /notebooks/embedding/data/raw
        wget https://github.com/songys/Question_pair/raw/master/kor_Pair_test.csv -P /notebooks/embedding/data/raw
        echo "download blog data.."
        gdrive_download 1Few7-Mh3JypQN3rjnuXD8yAXrkxUwmjS /notebooks/embedding/data/processed/processed_blog.txt
        echo "make directories..."
        mkdir /notebooks/embedding/data/tokenized
        mkdir /notebooks/embedding/data/trained-models
        ;;
    dump-processed)
        echo "download processed data and models..."
        mkdir -p /notebooks/embedding/data
        cd /notebooks/embedding/data
        gdrive_download 1hscU5_f_1vXfbhHabNpqfnp8DU2ZWmcT /notebooks/embedding/data/processed.zip
        gdrive_download 1u80kFwKlkN8ds0JvQtn6an2dl9QY1rE1 /notebooks/embedding/data/sentence-embeddings.zip
        gdrive_download 1vXiJr0qy_qA-bX4TxmDVqx1VB7_jRcIQ /notebooks/embedding/data/tokenized.zip
        gdrive_download 1yDUcFNlDT8KYaLLpo26aDboTILMcNAp6 /notebooks/embedding/data/trained-models.zip
        gdrive_download 1yHGtccC2FV3_d6C6_Q4cozYSOgA7bG-e /notebooks/embedding/data/word-embeddings.zip
        ls /notebooks/embedding/data/*.zip | xargs -n1 unzip
        rm /notebooks/embedding/data/*.zip
        ;;
    process_wiki)
        echo "processing ko-wikipedia..."
        python preprocess/dump.py --preprocess_mode wiki \
            --input_path /notebooks/embedding/data/raw/kowiki-latest-pages-articles.xml.bz2 \
            --output_path /notebooks/embedding/data/processed/processed_wiki_ko.txt
        ;;
    process_navermovie)
        echo "processing naver movie corpus..."
        python preprocess/dump.py --preprocess_mode nsmc \
            --input_path /notebooks/embedding/data/raw/ratings.txt \
            --output_path /notebooks/embedding/data/processed/processed_ratings.txt \
            --with_label False
        python preprocess/dump.py --preprocess_mode nsmc \
            --input_path /notebooks/embedding/data/raw/ratings_train.txt \
            --output_path /notebooks/embedding/data/processed/processed_ratings_train.txt \
            --with_label True
        python preprocess/dump.py --preprocess_mode nsmc \
            --input_path /notebooks/embedding/data/raw/ratings_test.txt \
            --output_path /notebooks/embedding/data/processed/processed_ratings_test.txt \
            --with_label True
        ;;
    process_korsquad)
        echo "processing KorSqaud corpus..."
        python preprocess/dump.py --preprocess_mode korsquad \
            --input_path /notebooks/embedding/data/raw/KorQuAD_v1.0_train.json \
            --output_path /notebooks/embedding/data/processed/processed_korsquad_train.txt
        python preprocess/dump.py --preprocess_mode korsquad \
            --input_path /notebooks/embedding/data/raw/KorQuAD_v1.0_dev.json \
            --output_path data/processed/processed_korsquad_dev.txt
        cat /notebooks/embedding/data/processed/processed_korsquad_train.txt /notebooks/embedding/data/processed/processed_korsquad_dev.txt > /notebooks/embedding/data/processed/processed_korsquad.txt
        rm /notebooks/embedding/data/processed/processed_korsquad_*.txt
        ;;
    mecab_tokenize)
        echo "mecab, tokenizing..."
        python preprocess/supervised_nlputils.py --tokenizer mecab \
            --input_path /notebooks/embedding/data/processed/processed_wiki_ko.txt \
            --output_path data/tokenized/wiki_ko_mecab.txt
        python preprocess/supervised_nlputils.py --tokenizer mecab \
            --input_path /notebooks/embedding/data/processed/processed_ratings.txt \
            --output_path data/tokenized/ratings_mecab.txt
        python preprocess/supervised_nlputils.py --tokenizer mecab \
            --input_path /notebooks/embedding/data/processed/processed_korsquad.txt \
            --output_path data/tokenized/korsquad_mecab.txt
        ;;
    space_correct)
        echo "train & apply space correct..."
        python preprocess/unsupervised_nlputils.py --preprocess_mode train_space \
            --input_path /notebooks/embedding/data/processed/processed_ratings.txt \
            --model_path /notebooks/embedding/data/trained-models/space-correct.model
        python preprocess/unsupervised_nlputils.py --preprocess_mode apply_space_correct \
            --input_path /notebooks/embedding/data/processed/processed_ratings.txt \
            --model_path /notebooks/embedding/data/trained-models/space-correct.model \
            --output_path /notebooks/embedding/data/processed/corrected_ratings_corpus.txt \
            --with_label False
        python preprocess/unsupervised_nlputils.py --preprocess_mode apply_space_correct \
            --input_path /notebooks/embedding/data/processed/processed_ratings_train.txt \
            --model_path /notebooks/embedding/data/trained-models/space-correct.model \
            --output_path /notebooks/embedding/data/processed/corrected_ratings_train.txt \
            --with_label True
        python preprocess/unsupervised_nlputils.py --preprocess_mode apply_space_correct \
            --input_path /notebooks/embedding/data/processed/processed_ratings_test.txt \
            --model_path /notebooks/embedding/data/trained-models/space-correct.model \
            --output_path /notebooks/embedding/data/processed/corrected_ratings_test.txt \
            --with_label True
        ;;
    soy_tokenize)
        echo "soynlp, LTokenizing..."
        python preprocess/unsupervised_nlputils.py --preprocess_mode compute_soy_word_score \
            --input_path /notebooks/embedding/data/processed/corrected_ratings_corpus.txt \
            --model_path /notebooks/embedding/data/trained-models/soyword.model
        python preprocess/unsupervised_nlputils.py --preprocess_mode soy_tokenize \
            --input_path /notebooks/embedding/data/processed/corrected_ratings_corpus.txt \
            --model_path /notebooks/embedding/data/trained-models/soyword.model \
            --output_path /notebooks/embedding/data/tokenized/ratings_soynlp.txt
        ;;
    komoran_tokenize)
        echo "komoran, tokenizing..."
        python preprocess/supervised_nlputils.py --tokenizer komoran \
            --input_path /notebooks/embedding/data/processed/corrected_ratings_corpus.txt \
            --output_path /notebooks/embedding/data/tokenized/ratings_komoran.txt
        ;;
    okt_tokenize)
        echo "okt, tokenizing..."
        python preprocess/supervised_nlputils.py --tokenizer okt \
            --input_path /notebooks/embedding/data/processed/corrected_ratings_corpus.txt \
            --output_path /notebooks/embedding/data/tokenized/ratings_okt.txt
        ;;
    hannanum_tokenize)
        echo "hannanum, tokenizing..."
        python preprocess/supervised_nlputils.py --tokenizer hannanum \
            --input_path /notebooks/embedding/data/processed/corrected_ratings_corpus.txt \
            --output_path /notebooks/embedding/data/tokenized/ratings_hannanum.txt
        ;;
    khaiii_tokenize)
        echo "khaiii, tokenizing..."
        python preprocess/supervised_nlputils.py --tokenizer khaiii \
            --input_path /notebooks/embedding/data/processed/corrected_ratings_corpus.txt \
            --output_path /notebooks/embedding/data/tokenized/ratings_khaiii.txt
        ;;
    sentencepiece)
        echo "processing sentencepiece..."
        cd /notebooks/embedding/data/trained-models
        spm_train --input=/notebooks/embedding/data/processed/corrected_ratings_corpus.txt --model_prefix=sentpiece --vocab_size=50000
        cd /notebooks/embedding
        python preprocess/unsupervised_nlputils.py --preprocess_mode process_sp_vocab \
            --input_path /notebooks/embedding/data/trained-models/sentpiece.vocab \
            --vocab_path /notebooks/embedding/data/trained-models/processed_sentpiece.vocab
        python preprocess/unsupervised_nlputils.py --preprocess_mode sentencepiece_tokenize \
            --vocab_path /notebooks/embedding/data/trained-models/processed_sentpiece.vocab \
            --input_path /notebooks/embedding/data/processed/corrected_ratings_corpus.txt \
            --output_path /notebooks/embedding/data/tokenized/ratings_sentpiece.txt
        ;;
esac