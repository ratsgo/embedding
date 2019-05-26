#!/usr/bin/env bash

COMMAND=$1

case $COMMAND in
    process-nsmc)
        echo "process nsmc raw json.."
        python preprocess/dump.py nsmc-json /Users/david/works/nsmc/raw data/review_movieid.txt True
        ;;
    lsa-tfidf)
        echo "latent semantic analysis with tf-idf matrix..."
        python models/sent_utils.py latent_semantic_analysis data/blog.txt data/lsa-tfidf.vecs
        ;;
    doc2vec)
        echo "doc2vec..."
        python models/sent_utils.py doc2vec data/review_movieid.txt data/doc2vec.vecs
        ;;
    lda)
        echo "latent_dirichlet_allocation..."
        python models/sent_utils.py latent_dirichlet_allocation data/corrected_ratings_corpus.txt data/lda
        ;;
    train-elmo)
        echo "train ELMo..."
        python models/sent_utils.py construct_elmo_vocab data/corpus_mecab.txt data/elmo-vocab.txt
        # options.json
        scp -P 30800 models/train_elmo.py ratsgo@112.217.184.162:~/bilm-tf
        scp -P 30800 models/bilm/* ratsgo@112.217.184.162:~/bilm-tf/bilm
        scp -P 30800 data/corpus_mecab.txt ratsgo@112.217.184.162:~/data
        scp -P 30800 data/elmo-vocab.txt ratsgo@112.217.184.162:~/data
        # @workstation
        mkdir data/traindata
        split -l 20000 data/corpus_mecab.txt data/traindata/data_
        cd ~/bilm-tf
        source ~/tf120/bin/activate
        export CUDA_VISIBLE_DEVICES=0,1
        export LC_CTYPE=C.UTF-8
        python3.6 train_elmo.py \
            --train_prefix='/home/ratsgo/data/traindata/*' \
            --vocab_file /home/ratsgo/data/elmo-vocab.txt \
            --save_dir /home/ratsgo/elmo-model
        ;;
    dump-elmo)
        echo "dump pretrained ELMo weights..."
        # @workstation
        source ~/tf120/bin/activate
        export CUDA_VISIBLE_DEVICES=0
        python3.6 models/sent_utils.py dump_elmo_weights data/elmo/ckpt data/elmo/elmo.model
        ;;
    tune-elmo)
        echo "tune ELMo..."
        # @local
        scp -P 30800 data/ratings_train.txt.elmo.tokenized ratsgo@112.217.184.162:~/embedding/data
        scp -P 30800 data/ratings_test.txt.elmo.tokenized ratsgo@112.217.184.162:~/embedding/data
        scp -P 30800 data/elmo-vocab.txt ratsgo@112.217.184.162:~/embedding/data
        # @workstation
        source ~/tf120/bin/activate
        export CUDA_VISIBLE_DEVICES=0
        export LC_CTYPE=C.UTF-8
        nohup sh -c "python3.6 models/tune_utils.py elmo data/ratings_train.txt data/ratings_test.txt data/elmo-vocab.txt data/elmo/elmo.model data/elmo/ckpt/options.json data/elmo" > elmo.log &
        ;;
    dump-bert)
        echo "dump pretrained BERT weights..."
        wget https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip
        mv multi_cased_L-12_H-768_A-12.zip data/bert
        cd data/bert
        unzip multi_cased_L-12_H-768_A-12.zip
        ;;
    tune-bert)
        echo "tune BERT..."
        # @local
        scp -P 30800 data/corrected_ratings_train.txt ratsgo@112.217.184.162:~/embedding/data
        scp -P 30800 data/corrected_ratings_test.txt ratsgo@112.217.184.162:~/embedding/data
        scp -P 30800 data/bert/multi_cased_L-12_H-768_A-12/* ratsgo@112.217.184.162:~/embedding/data/bert/multi_cased_L-12_H-768_A-12
        # @workstation
        source ~/tf120/bin/activate
        export CUDA_VISIBLE_DEVICES=1
        export LC_CTYPE=C.UTF-8
        nohup sh -c "python3.6 models/tune_utils.py bert data/corrected_ratings_train.txt data/corrected_ratings_test.txt data/bert/multi_cased_L-12_H-768_A-12/vocab.txt data/bert/multi_cased_L-12_H-768_A-12/bert_model.ckpt data/bert/multi_cased_L-12_H-768_A-12/bert_config.json data/bert" > bert.log &
        ;;
        ;;
esac