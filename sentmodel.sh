#!/usr/bin/env bash

COMMAND=$1

case $COMMAND in
    process-nsmc)
        echo "process nsmc raw json.."
        cd /notebooks
        git clone https://github.com/e9t/nsmc.git
        python preprocess/dump.py nsmc-json /notebooks/nsmc/raw /notebooks/embedding/data/processed/processed_review_movieid.txt True
        ;;
    lsa-tfidf)
        echo "latent semantic analysis with tf-idf matrix..."
        mkdir -p /notebooks/embedding/data/sentence-embeddings/lsa-tfidf
        python models/sent_utils.py latent_semantic_analysis /notebooks/embedding/data/processed/processed_blog.txt /notebooks/embedding/data/sentence-embeddings/lsa-tfidf/lsa-tfidf.vecs
        ;;
    doc2vec)
        echo "train doc2vec model..."
        mkdir -p /notebooks/embedding/data/sentence-embeddings/doc2vec
        python models/sent_utils.py doc2vec /notebooks/embedding/data/processed/processed_review_movieid.txt /notebooks/embedding/data/sentence-embeddings/doc2vec/doc2vec.model
        ;;
    lda)
        echo "latent_dirichlet_allocation..."
        mkdir -p /notebooks/embedding/data/sentence-embeddings/lda
        python models/sent_utils.py latent_dirichlet_allocation /notebooks/embedding/data/processed/corrected_ratings_corpus.txt /notebooks/embedding/data/sentence-embeddings/lda/lda
        ;;
    pretrain-elmo)
        echo "pretrain ELMo..."
        mkdir -p /notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt/traindata
        cat /notebooks/embedding/data/tokenized/wiki_ko_mecab.txt /notebooks/embedding/data/tokenized/ratings_mecab.txt /notebooks/embedding/data/tokenized/korsquad_mecab.txt > /notebooks/embedding/data/tokenized/corpus_mecab.txt
        export LC_CTYPE=C.UTF-8
        python models/sent_utils.py construct_elmo_vocab /notebooks/embedding/data/tokenized/corpus_mecab.txt /notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt/elmo-vocab.txt
        split -l 100000 /notebooks/embedding/data/tokenized/corpus_mecab.txt /notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt/traindata/data_
        nohup sh -c "python models/train_elmo.py \
            --train_prefix='/notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt/traindata/*' \
            --vocab_file /notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt/elmo-vocab.txt \
            --save_dir /notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt" > elmo-pretrain.log &
        ;;
    dump-pretrained-elmo)
        echo "dump pretrained ELMo weights..."
        python models/sent_utils.py dump_elmo_weights /notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt /notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt/elmo.model
        ;;
    tune-elmo)
        echo "tune ELMo..."
        export LC_CTYPE=C.UTF-8
        nohup sh -c "python models/tune_utils.py elmo \
                      /notebooks/embedding/data/processed/processed_ratings_train.txt \
                      /notebooks/embedding/data/processed/processed_ratings_test.txt \
                      /notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt/elmo-vocab.txt \
                      /notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt/elmo.model \
                      /notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt/options.json \
                      /notebooks/embedding/data/sentence-embeddings/elmo/tune-ckpt" > elmo-tune.log &
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
        nohup sh -c "python models/tune_utils.py bert /notebooks/embedding/data/processed/processed_ratings_train.txt /notebooks/embedding/data/processed/processed_ratings_test.txt /notebooks/embedding/data/sentence-embeddings/bert/multi_cased_L-12_H-768_A-12/vocab.txt /notebooks/embedding/data/sentence-embeddings/bert/multi_cased_L-12_H-768_A-12/bert_model.ckpt /notebooks/embedding/data/sentence-embeddings/bert/multi_cased_L-12_H-768_A-12/bert_config.json /notebooks/embedding/data/sentence-embeddings/bert/tune-ckpt" > bert.log &
        ;;
esac