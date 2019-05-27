#!/usr/bin/env bash

COMMAND=$1

case $COMMAND in
    process-nsmc)
        echo "process nsmc raw json.."
        cd /notebooks
        git clone https://github.com/e9t/nsmc.git
        python preprocess/dump.py --preprocess_mode nsmc-json \
            --input_path /notebooks/nsmc/raw \
            --output_path /notebooks/embedding/data/processed/processed_review_movieid.txt \
            --with_label True
        ;;
    lsa-tfidf)
        echo "latent semantic analysis with tf-idf matrix..."
        mkdir -p /notebooks/embedding/data/sentence-embeddings/lsa-tfidf
        python models/sent_utils.py --method latent_semantic_analysis \
            --input_path /notebooks/embedding/data/processed/processed_blog.txt \
            --output_path /notebooks/embedding/data/sentence-embeddings/lsa-tfidf/lsa-tfidf.vecs
        ;;
    doc2vec)
        echo "train doc2vec model..."
        mkdir -p /notebooks/embedding/data/sentence-embeddings/doc2vec
        python models/sent_utils.py --method doc2vec \
            --input_path /notebooks/embedding/data/processed/processed_review_movieid.txt \
            --output_path /notebooks/embedding/data/sentence-embeddings/doc2vec/doc2vec.model
        ;;
    lda)
        echo "latent_dirichlet_allocation..."
        mkdir -p /notebooks/embedding/data/sentence-embeddings/lda
        python models/sent_utils.py --method latent_dirichlet_allocation \
            --input_path /notebooks/embedding/data/processed/corrected_ratings_corpus.txt \
            --output_path /notebooks/embedding/data/sentence-embeddings/lda/lda
        ;;
    pretrain-elmo)
        echo "pretrain ELMo..."
        mkdir -p /notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt/traindata
        cat /notebooks/embedding/data/tokenized/wiki_ko_mecab.txt /notebooks/embedding/data/tokenized/ratings_mecab.txt /notebooks/embedding/data/tokenized/korsquad_mecab.txt > /notebooks/embedding/data/tokenized/corpus_mecab.txt
        export LC_CTYPE=C.UTF-8
        python models/sent_utils.py --method construct_elmo_vocab \
            --input_path /notebooks/embedding/data/tokenized/corpus_mecab.txt \
            --output_path /notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt/elmo-vocab.txt
        split -l 100000 /notebooks/embedding/data/tokenized/corpus_mecab.txt /notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt/traindata/data_
        nohup sh -c "python models/train_elmo.py \
            --train_prefix='/notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt/traindata/*' \
            --vocab_file /notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt/elmo-vocab.txt \
            --save_dir /notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt \
            --n_gpus 1" > elmo-pretrain.log &
        ;;
    dump-pretrained-elmo)
        echo "dump pretrained ELMo weights..."
        python models/sent_utils.py --method dump_elmo_weights \
            --input_path /notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt \
            --output_path /notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt/elmo.model
        ;;
    tune-elmo)
        echo "tune ELMo..."
        export LC_CTYPE=C.UTF-8
        nohup sh -c "python models/tune_utils.py --model_name elmo \
                      --train_corpus_fname /notebooks/embedding/data/processed/processed_ratings_train.txt \
                      --test_corpus_fname /notebooks/embedding/data/processed/processed_ratings_test.txt \
                      --vocab_fname /notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt/elmo-vocab.txt \
                      --pretrain_model_fname /notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt/elmo.model \
                      --config_fname /notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt/options.json \
                      --model_save_path /notebooks/embedding/data/sentence-embeddings/elmo/tune-ckpt" > elmo-tune.log &
        ;;
    dump-pretrained-bert)
        echo "dump pretrained BERT weights..."
        wget https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip -O /notebooks/embedding/data/sentence-embeddings/bert/multi_cased_L-12_H-768_A-12.zip
        cd /notebooks/embedding/data/sentence-embeddings/bert
        unzip multi_cased_L-12_H-768_A-12.zip
        rm multi_cased_L-12_H-768_A-12.zip
        ;;
    tune-bert)
        echo "tune BERT..."
        export LC_CTYPE=C.UTF-8
        nohup sh -c "python models/tune_utils.py --model_name bert \
                      --train_corpus_fname /notebooks/embedding/data/processed/processed_ratings_train.txt \
                      --test_corpus_fname /notebooks/embedding/data/processed/processed_ratings_test.txt \
                      --vocab_fname /notebooks/embedding/data/sentence-embeddings/bert/multi_cased_L-12_H-768_A-12/vocab.txt \
                      --pretrain_model_fname /notebooks/embedding/data/sentence-embeddings/bert/multi_cased_L-12_H-768_A-12/bert_model.ckpt \
                      --config_fname /notebooks/embedding/data/sentence-embeddings/bert/multi_cased_L-12_H-768_A-12/bert_config.json \
                      --model_save_path /notebooks/embedding/data/sentence-embeddings/bert/tune-ckpt" > bert.log &
        ;;
esac