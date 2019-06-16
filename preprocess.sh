#!/usr/bin/env bash

COMMAND=$1
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
        echo "download KorQuAD data..."
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
    dump-word-embedding)
        echo "download word embeddings..."
        mkdir -p /notebooks/embedding/data
        cd /notebooks/embedding/data
        gdrive_download 1gpOaOl0BcUvYpgoOA2JpZY2z-BUhuBLX /notebooks/embedding/data/word-embeddings.zip
        unzip word-embeddings.zip
        rm word-embeddings.zip
        ;;
    dump-sentence-embedding)
        echo "download sentence embeddings..."
        mkdir -p /notebooks/embedding/data
        cd /notebooks/embedding/data
        gdrive_download 1y_58tgW4S9ujOrwUs9oUtLMM6MfGhctr /notebooks/embedding/data/sentence-embeddings.zip
        unzip sentence-embeddings.zip
        rm sentence-embeddings.zip
        ;;
    dump-tokenized)
        echo "download tokenized data..."
        mkdir -p /notebooks/embedding/data
        cd /notebooks/embedding/data
        gdrive_download 1QEdjvT0Jpqmz9F57ATmjy3i016kdcURq /notebooks/embedding/data/tokenized.zip
        unzip tokenized.zip
        rm tokenized.zip
        ;;
    dump-trained-models)
        echo "download trained models..."
        mkdir -p /notebooks/embedding/data
        cd /notebooks/embedding/data
        gdrive_download 1RBZ13ixJKQL3OozjgXWH4rDuOHGS1r8R /notebooks/embedding/data/trained-models.zip
        unzip trained-models.zip
        rm trained-models.zip
        ;;
    dump-processed)
        echo "download processed data..."
        mkdir -p /notebooks/embedding/data
        cd /notebooks/embedding/data
        gdrive_download 1oO5v6YqNlKTq0vWfjME3SiLXAYCMAmkc /notebooks/embedding/data/processed.zip
        unzip processed.zip
        rm processed.zip
        ;;
    process-wiki)
        echo "processing ko-wikipedia..."
        python preprocess/dump.py --preprocess_mode wiki \
            --input_path /notebooks/embedding/data/raw/kowiki-latest-pages-articles.xml.bz2 \
            --output_path /notebooks/embedding/data/processed/processed_wiki_ko.txt
        ;;
    process-navermovie)
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
    process-korquad)
        echo "processing KorQaAD corpus..."
        python preprocess/dump.py --preprocess_mode korquad \
            --input_path /notebooks/embedding/data/raw/KorQuAD_v1.0_train.json \
            --output_path /notebooks/embedding/data/processed/processed_korquad_train.txt
        python preprocess/dump.py --preprocess_mode korquad \
            --input_path /notebooks/embedding/data/raw/KorQuAD_v1.0_dev.json \
            --output_path data/processed/processed_korquad_dev.txt
        cat /notebooks/embedding/data/processed/processed_korquad_train.txt /notebooks/embedding/data/processed/processed_korquad_dev.txt > /notebooks/embedding/data/processed/processed_korquad.txt
        rm /notebooks/embedding/data/processed/processed_korquad_*.txt
        ;;
    mecab-tokenize)
        echo "mecab, tokenizing..."
        python preprocess/supervised_nlputils.py --tokenizer mecab \
            --input_path /notebooks/embedding/data/processed/processed_wiki_ko.txt \
            --output_path data/tokenized/wiki_ko_mecab.txt
        python preprocess/supervised_nlputils.py --tokenizer mecab \
            --input_path /notebooks/embedding/data/processed/processed_ratings.txt \
            --output_path data/tokenized/ratings_mecab.txt
        python preprocess/supervised_nlputils.py --tokenizer mecab \
            --input_path /notebooks/embedding/data/processed/processed_korquad.txt \
            --output_path data/tokenized/korquad_mecab.txt
        ;;
    process-jamo)
        echo "processing jamo sentences..."
        python preprocess/unsupervised_nlputils.py --preprocess_mode jamo \
            --input_path /notebooks/embedding/data/tokenized/corpus_mecab.txt \
            --output_path /notebooks/embedding/data/tokenized/corpus_mecab_jamo.txt
        ;;
    space-correct)
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
    soy-tokenize)
        echo "soynlp, LTokenizing..."
        python preprocess/unsupervised_nlputils.py --preprocess_mode compute_soy_word_score \
            --input_path /notebooks/embedding/data/processed/corrected_ratings_corpus.txt \
            --model_path /notebooks/embedding/data/trained-models/soyword.model
        python preprocess/unsupervised_nlputils.py --preprocess_mode soy_tokenize \
            --input_path /notebooks/embedding/data/processed/corrected_ratings_corpus.txt \
            --model_path /notebooks/embedding/data/trained-models/soyword.model \
            --output_path /notebooks/embedding/data/tokenized/ratings_soynlp.txt
        ;;
    komoran-tokenize)
        echo "komoran, tokenizing..."
        python preprocess/supervised_nlputils.py --tokenizer komoran \
            --input_path /notebooks/embedding/data/processed/corrected_ratings_corpus.txt \
            --output_path /notebooks/embedding/data/tokenized/ratings_komoran.txt
        ;;
    okt-tokenize)
        echo "okt, tokenizing..."
        python preprocess/supervised_nlputils.py --tokenizer okt \
            --input_path /notebooks/embedding/data/processed/corrected_ratings_corpus.txt \
            --output_path /notebooks/embedding/data/tokenized/ratings_okt.txt
        ;;
    hannanum-tokenize)
        echo "hannanum, tokenizing..."
        python preprocess/supervised_nlputils.py --tokenizer hannanum \
            --input_path /notebooks/embedding/data/processed/corrected_ratings_corpus.txt \
            --output_path /notebooks/embedding/data/tokenized/ratings_hannanum.txt
        ;;
    khaiii-tokenize)
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
    mecab-user-dic)
        echo "insert mecab user dictionary..."
        cp -f preprocess/mecab-user-dic.csv /tmp/mecab-ko-dic-2.1.1-20180720/user-dic/nnp.csv
        bash /tmp/mecab-ko-dic-2.1.1-20180720/tools/add-userdic.sh
        cd /tmp/mecab-ko-dic-2.1.1-20180720/user-dic
        make install
        cd /notebooks/embedding
        ;;
esac