#!/usr/bin/env bash

COMMAND=$1
function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}

case $COMMAND in
    process-nsmc)
        echo "process nsmc raw json.."
        cd /notebooks
        git clone https://github.com/e9t/nsmc.git
        cd /notebooks/embedding
        python preprocess/dump.py --preprocess_mode nsmc-json \
            --input_path /notebooks/nsmc/raw \
            --output_path /notebooks/embedding/data/processed/processed_review_movieid.txt \
            --with_label True
        ;;
    lsa-tfidf)
        echo "latent semantic analysis with tf-idf matrix..."
        python models/sent_utils.py --method latent_semantic_analysis \
            --input_path /notebooks/embedding/data/processed/processed_blog.txt \
            --output_path /notebooks/embedding/data/sentence-embeddings/lsa-tfidf/lsa-tfidf.vecs
        ;;
    doc2vec)
        echo "train doc2vec model..."
        python models/sent_utils.py --method doc2vec \
            --input_path /notebooks/embedding/data/processed/processed_review_movieid.txt \
            --output_path /notebooks/embedding/data/sentence-embeddings/doc2vec/doc2vec.model
        ;;
    lda)
        echo "latent_dirichlet_allocation..."
        python models/sent_utils.py --method latent_dirichlet_allocation \
            --input_path /notebooks/embedding/data/processed/corrected_ratings_corpus.txt \
            --output_path /notebooks/embedding/data/sentence-embeddings/lda/lda
        ;;
    tune-random)
        echo "tune random init with Bi-LSTM attention model..."
        nohup sh -c "python models/tune_utils.py --model_name word \
                      --train_corpus_fname /notebooks/embedding/data/processed/processed_ratings_train.txt \
                      --test_corpus_fname /notebooks/embedding/data/processed/processed_ratings_test.txt \
                      --embedding_name random \
                      --model_save_path /notebooks/embedding/data/word-embeddings/random-tune" > tune-random.log &
        ;;
    tune-word2vec)
        echo "tune word2vec with Bi-LSTM attention model..."
        nohup sh -c "python models/tune_utils.py --model_name word \
                      --train_corpus_fname /notebooks/embedding/data/processed/processed_ratings_train.txt \
                      --test_corpus_fname /notebooks/embedding/data/processed/processed_ratings_test.txt \
                      --embedding_name word2vec \
                      --embedding_fname /notebooks/embedding/data/word-embeddings/word2vec/word2vec \
                      --model_save_path /notebooks/embedding/data/word-embeddings/word2vec-tune" > tune-word2vec.log &
        ;;
    tune-glove)
        echo "tune glove with Bi-LSTM attention model..."
        nohup sh -c "python models/tune_utils.py --model_name word \
                      --train_corpus_fname /notebooks/embedding/data/processed/processed_ratings_train.txt \
                      --test_corpus_fname /notebooks/embedding/data/processed/processed_ratings_test.txt \
                      --embedding_name glove \
                      --embedding_fname /notebooks/embedding/data/word-embeddings/glove/glove.txt \
                      --model_save_path /notebooks/embedding/data/word-embeddings/glove-tune" > tune-glove.log &
        ;;
    tune-ft)
        echo "tune fasttext with Bi-LSTM attention model..."
        nohup sh -c "python models/tune_utils.py --model_name word \
                      --train_corpus_fname /notebooks/embedding/data/processed/processed_ratings_train.txt \
                      --test_corpus_fname /notebooks/embedding/data/processed/processed_ratings_test.txt \
                      --embedding_name fasttext \
                      --embedding_fname /notebooks/embedding/data/word-embeddings/fasttext/fasttext.vec \
                      --model_save_path /notebooks/embedding/data/word-embeddings/fasttext-tune" > tune-ft.log &
        ;;
    tune-swivel)
        echo "tune swivel with Bi-LSTM attention model..."
        nohup sh -c "python models/tune_utils.py --model_name word \
                      --train_corpus_fname /notebooks/embedding/data/processed/processed_ratings_train.txt \
                      --test_corpus_fname /notebooks/embedding/data/processed/processed_ratings_test.txt \
                      --embedding_name swivel \
                      --embedding_fname /notebooks/embedding/data/word-embeddings/swivel/row_embedding.tsv \
                      --model_save_path /notebooks/embedding/data/word-embeddings/swivel-tune" > tune-swivel.log &
        ;;
    pretrain-elmo)
        echo "pretrain ELMo..."
        mkdir -p /notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt/traindata
        cat /notebooks/embedding/data/tokenized/wiki_ko_mecab.txt /notebooks/embedding/data/tokenized/ratings_mecab.txt /notebooks/embedding/data/tokenized/korquad_mecab.txt > /notebooks/embedding/data/tokenized/corpus_mecab.txt
        split -l 100000 /notebooks/embedding/data/tokenized/corpus_mecab.txt /notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt/traindata/data_
        python models/sent_utils.py --method construct_elmo_vocab \
            --input_path /notebooks/embedding/data/tokenized/corpus_mecab.txt \
            --output_path /notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt/elmo-vocab.txt
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
    download-pretrained-elmo)
        echo "download pretrained ELMo weights..."
        mkdir -p /notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt
        gdrive_download 1go2JtVeYBOjkBCWJWk8inkSpVg7VfFVp /notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt/elmo.zip
        cd /notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt
        unzip elmo.zip
        rm elmo.zip
        cd /notebooks/embedding
        ;;
    tune-elmo)
        echo "tune ELMo..."
        nohup sh -c "python models/tune_utils.py --model_name elmo \
                      --train_corpus_fname /notebooks/embedding/data/processed/processed_ratings_train.txt \
                      --test_corpus_fname /notebooks/embedding/data/processed/processed_ratings_test.txt \
                      --vocab_fname /notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt/elmo-vocab.txt \
                      --pretrain_model_fname /notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt/elmo.model \
                      --config_fname /notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt/options.json \
                      --model_save_path /notebooks/embedding/data/sentence-embeddings/elmo/tune-ckpt" > elmo-tune.log &
        ;;
    pretrain-bert)
        echo "precess documents..."
        mkdir -p /notebooks/embedding/data/sentence-embeddings/pretrain-data
        python preprocess/dump.py --preprocess_mode process-documents \
            --input_path /notebooks/embedding/data/processed/corrected_ratings_corpus.txt \
            --output_path /notebooks/embedding/data/processed/pretrain.txt
        split -l 300000 /notebooks/embedding/data/processed/pretrain.txt /notebooks/embedding/data/sentence-embeddings/pretrain-data/data_
        echo "processing BERT vocabulary..."
        mkdir -p /notebooks/embedding/data/sentence-embeddings/bert/pretrain-ckpt
        python preprocess/unsupervised_nlputils.py --preprocess_mode make_bert_vocab \
            --input_path /notebooks/embedding/data/processed/pretrain.txt \
            --vocab_path /notebooks/embedding/data/sentence-embeddings/bert/pretrain-ckpt/vocab.txt
        mv sentpiece* /notebooks/embedding/data/processed
        echo "preprocess corpus..."
        mkdir -p /notebooks/embedding/data/sentence-embeddings/bert/pretrain-ckpt/traindata
        python models/bert/create_pretraining_data.py \
            --input_file=/notebooks/embedding/data/sentence-embeddings/pretrain-data/* \
            --output_file=/notebooks/embedding/data/sentence-embeddings/bert/pretrain-ckpt/traindata/tfrecord \
            --vocab_file=/notebooks/embedding/data/sentence-embeddings/bert/pretrain-ckpt/vocab.txt \
            --do_lower_case=False \
            --max_seq_length=128 \
            --max_predictions_per_seq=20 \
            --masked_lm_prob=0.15 \
            --random_seed=7 \
            --dupe_factor=5
        echo "pretrain fresh BERT..."
        gdrive_download 1DEpdPRJc-kjmeU_5pgMPzTOuH8qrwIzY /notebooks/embedding/data/sentence-embeddings/bert/pretrain-ckpt/bert_config.json
        nohup sh -c "python models/bert/run_pretraining.py \
                      --input_file=/notebooks/embedding/data/sentence-embeddings/bert/pretrain-ckpt/traindata/tfrecord* \
                      --output_dir=/notebooks/embedding/data/sentence-embeddings/bert/pretrain-ckpt \
                      --do_train=True \
                      --do_eval=True \
                      --bert_config_file=/notebooks/embedding/data/sentence-embeddings/bert/pretrain-ckpt/bert_config.json \
                      --train_batch_size=32 \
                      --max_seq_length=128 \
                      --max_predictions_per_seq=20 \
                      --learning_rate=2e-5" > bert-pretrain.log &
        ;;
    download-pretrained-bert)
        echo "download pretrained BERT weights..."
        mkdir -p /notebooks/embedding/data/sentence-embeddings/bert/pretrain-ckpt
        gdrive_download 1DEpdPRJc-kjmeU_5pgMPzTOuH8qrwIzY /notebooks/embedding/data/sentence-embeddings/bert/pretrain-ckpt/bert_config.json
        gdrive_download 12cCImHAM97lXb427vCl_3MXOY7bxNlYe /notebooks/embedding/data/sentence-embeddings/bert/pretrain-ckpt/bert_model.ckpt.data-00000-of-00001
        gdrive_download 10jD8gN94Vr_5XMftheJB7n0IBm-pjdwd /notebooks/embedding/data/sentence-embeddings/bert/pretrain-ckpt/bert_model.ckpt.index
        gdrive_download 1pLNR2xL17HCLD3GmWCLls7a9xhhDIdw2 /notebooks/embedding/data/sentence-embeddings/bert/pretrain-ckpt/bert_model.ckpt.meta
        gdrive_download 1LkyTFPeoTvWoO5XP0bDi3Af53XPCLE59 /notebooks/embedding/data/sentence-embeddings/bert/pretrain-ckpt/vocab.txt
        ;;
    tune-bert)
        echo "tune BERT..."
        nohup sh -c "python models/tune_utils.py --model_name bert \
                      --train_corpus_fname /notebooks/embedding/data/processed/processed_ratings_train.txt \
                      --test_corpus_fname /notebooks/embedding/data/processed/processed_ratings_test.txt \
                      --vocab_fname /notebooks/embedding/data/sentence-embeddings/bert/pretrain-ckpt/vocab.txt \
                      --pretrain_model_fname /notebooks/embedding/data/sentence-embeddings/bert/pretrain-ckpt/bert_model.ckpt \
                      --config_fname /notebooks/embedding/data/sentence-embeddings/bert/pretrain-ckpt/bert_config.json \
                      --model_save_path /notebooks/embedding/data/sentence-embeddings/bert/tune-ckpt" > bert-tune.log &
        ;;
    pretrain-xlnet)
        echo "precess documents..."
        mkdir -p /notebooks/embedding/data/sentence-embeddings/pretrain-data
        python preprocess/dump.py --preprocess_mode process-documents \
            --input_path /notebooks/embedding/data/processed/corrected_ratings_corpus.txt \
            --output_path /notebooks/embedding/data/processed/pretrain.txt
        split -l 300000 /notebooks/embedding/data/processed/pretrain.txt /notebooks/embedding/data/sentence-embeddings/pretrain-data/data_
        echo "construct XLNet vocabulary..."
        mkdir -p /notebooks/embedding/data/sentence-embeddings/xlnet/pretrain-ckpt
        python preprocess/unsupervised_nlputils.py --preprocess_mode make_xlnet_vocab \
            --input_path /notebooks/embedding/data/processed/pretrain.txt \
            --vocab_path /notebooks/embedding/data/sentence-embeddings/xlnet/pretrain-ckpt/sp10m.cased.v3
        echo "preprocess corpus..."
        cd models/xlnet
        python data_utils.py --bsz_per_host=16 \
	                         --num_core_per_host=1 \
	                         --seq_len=256 \
	                         --reuse_len=128 \
	                         --input_glob=/notebooks/embedding/data/sentence-embeddings/pretrain-data/* \
	                         --save_dir=/notebooks/embedding/data/sentence-embeddings/xlnet/pretrain-ckpt \
	                         --num_passes=10 \
	                         --bi_data=True \
	                         --sp_path=/notebooks/embedding/data/sentence-embeddings/xlnet/pretrain-ckpt/sp10m.cased.v3.model \
	                         --mask_alpha=6 \
	                         --mask_beta=1 \
	                         --num_predict=45
	    python train_gpu.py --record_info_dir=/notebooks/embedding/data/sentence-embeddings/xlnet/pretrain-ckpt/tfrecords \
                            --model_dir=/notebooks/embedding/data/sentence-embeddings/xlnet/pretrain-ckpt \
                            --train_batch_size=16 \
                            --seq_len=256 \
                            --reuse_len=128 \
                            --mem_len=192 \
                            --perm_size=128 \
                            --n_layer=3 \
                            --d_model=512 \
                            --d_embed=512 \
                            --n_head=8 \
                            --d_head=32 \
                            --d_inner=2048 \
                            --uncased=True \
                            --untie_r=True \
                            --mask_alpha=6 \
                            --mask_beta=1 \
                            --num_predict=45 \
                            --save_steps=10000
        ;;
    download-pretrained-xlnet)
        echo "download pretrained xlnet..."
        mkdir -p /notebooks/embedding/data/sentence-embeddings/xlnet/pretrain-ckpt
        gdrive_download 1fzK6bh1dXfBIceH7DzFxQ_1vJSNPDeuX /notebooks/embedding/data/sentence-embeddings/xlnet/pretrain-ckpt/xlnet_config.json
        gdrive_download 1o72uZ9B3f887F1QKza6WCdPvc8g7DL8j /notebooks/embedding/data/sentence-embeddings/xlnet/pretrain-ckpt/xlnet_model.ckpt.data-00000-of-00001
        gdrive_download 1qP-imkmBbC4BLkBSk2_fjOuCEllNzzSX /notebooks/embedding/data/sentence-embeddings/xlnet/pretrain-ckpt/xlnet_model.ckpt.index
        gdrive_download 1bu1lOC_WtTqwLM6P70zjlk_hsS3SiwBV /notebooks/embedding/data/sentence-embeddings/xlnet/pretrain-ckpt/xlnet_model.ckpt.meta
        gdrive_download 1kIYl_KFwuqbwfvLbI-bU8mlteMM5Ond- /notebooks/embedding/data/sentence-embeddings/xlnet/pretrain-ckpt/sp10m.cased.v3.model
        ;;
    tune-xlnet)
        echo "tune XLNet..."
        nohup sh -c "python models/tune_utils.py --model_name xlnet \
                      --train_corpus_fname /notebooks/embedding/data/processed/processed_ratings_train.txt \
                      --test_corpus_fname /notebooks/embedding/data/processed/processed_ratings_test.txt \
                      --vocab_fname /notebooks/embedding/data/sentence-embeddings/xlnet/pretrain-ckpt/sp10m.cased.v3.model \
                      --pretrain_model_fname /notebooks/embedding/data/sentence-embeddings/xlnet/pretrain-ckpt/xlnet_model.ckpt \
                      --config_fname /notebooks/embedding/data/sentence-embeddings/xlnet/pretrain-ckpt/xlnet_config.json \
                      --model_save_path /notebooks/embedding/data/sentence-embeddings/xlnet/tune-ckpt" > xlnet-tune.log &
        ;;
esac