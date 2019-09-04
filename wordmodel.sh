#!/usr/bin/env bash

COMMAND=$1
case $COMMAND in
    merge)
        echo "merge datasets.."
        cat /notebooks/embedding/data/tokenized/wiki_ko_mecab.txt /notebooks/embedding/data/tokenized/ratings_mecab.txt /notebooks/embedding/data/tokenized/korquad_mecab.txt > /notebooks/embedding/data/tokenized/corpus_mecab.txt
        cat /notebooks/embedding/data/tokenized/ratings_mecab.txt /notebooks/embedding/data/tokenized/korquad_mecab.txt > /notebooks/embedding/data/tokenized/for-lsa-mecab.txt
        ;;
    lsa)
        echo "latent semantic analysis..."
        python models/word_utils.py --method latent_semantic_analysis \
            --input_path /notebooks/embedding/data/tokenized/for-lsa-mecab.txt \
            --output_path /notebooks/embedding/data/word-embeddings/lsa/lsa
        ;;
    word2vec)
        echo "train word2vec word embedding..."
        python models/word_utils.py --method train_word2vec \
            --input_path /notebooks/embedding/data/tokenized/corpus_mecab.txt \
            --output_path /notebooks/embedding/data/word-embeddings/word2vec/word2vec
        python models/word_utils.py --method train_word2vec \
            --input_path /notebooks/embedding/data/tokenized/for-lsa-mecab.txt \
            --output_path /notebooks/embedding/data/word-embeddings/word2vec/word2vec-lsa
        ;;
    glove)
        echo "train glove word embedding..."
        mkdir -p /notebooks/embedding/data/word-embeddings/glove
        /notebooks/embedding/models/glove/build/vocab_count -min-count 5 -verbose 2 < /notebooks/embedding/data/tokenized/corpus_mecab.txt > /notebooks/embedding/data/word-embeddings/glove/glove.vocab
        /notebooks/embedding/models/glove/build/cooccur -memory 10.0 -vocab-file /notebooks/embedding/data/word-embeddings/glove/glove.vocab -verbose 2 -window-size 15 < /notebooks/embedding/data/tokenized/corpus_mecab.txt > /notebooks/embedding/data/word-embeddings/glove/glove.cooc
        /notebooks/embedding/models/glove/build/shuffle -memory 10.0 -verbose 2 < /notebooks/embedding/data/word-embeddings/glove/glove.cooc > /notebooks/embedding/data/word-embeddings/glove/glove.shuf
        /notebooks/embedding/models/glove/build/glove -save-file /notebooks/embedding/data/word-embeddings/glove/glove.vecs -threads 4 -input-file /notebooks/embedding/data/word-embeddings/glove/glove.shuf -x-max 10 -iter 15 -vector-size 100 -binary 2 -vocab-file /notebooks/embedding/data/word-embeddings/glove/glove.vocab -verbose 2
        ;;
    fasttext)
        echo "train fasttext word embedding..."
        mkdir -p /notebooks/embedding/data/word-embeddings/fasttext
        /notebooks/embedding/models/fastText/fasttext skipgram -input /notebooks/embedding/data/tokenized/corpus_mecab.txt -output /notebooks/embedding/data/word-embeddings/fasttext/fasttext
        ;;
    fasttext-jamo)
        echo "train fasttext jamo embedding..."
        mkdir -p /notebooks/embedding/data/word-embeddings/fasttext-jamo
        /notebooks/embedding/models/fastText/fasttext skipgram -input /notebooks/embedding/data/tokenized/corpus_mecab_jamo.txt -output /notebooks/embedding/data/word-embeddings/fasttext-jamo/fasttext-jamo
        ;;
    swivel)
        echo "train swivel word embedding..."
        mkdir -p /notebooks/embedding/data/word-embeddings/swivel
        /notebooks/embedding/models/swivel/fastprep --input /notebooks/embedding/data/tokenized/corpus_mecab.txt --output_dir /notebooks/embedding/data/word-embeddings/swivel/swivel.data
        python /notebooks/embedding/models/swivel/swivel.py --input_base_path /notebooks/embedding/data/word-embeddings/swivel/swivel.data --output_base_path /notebooks/embedding/data/word-embeddings/swivel --dim 100
        ;;
    cbow)
        echo "evaluate weighted embeddings..."
        # word2vec - original
        python /notebooks/embedding/models/word_utils.py --train_corpus_path /notebooks/embedding/data/processed/processed_ratings_train.txt --test_corpus_path /notebooks/embedding/data/processed/processed_ratings_test.txt --embedding_path /notebooks/embedding/data/word-embeddings/word2vec/word2vec --output_path /notebooks/embedding/data/word-embeddings/cbow/word2vec --embedding_name word2vec --method cbow --is_weighted False
        # word2vec - weighted
        python /notebooks/embedding/models/word_utils.py --train_corpus_path /notebooks/embedding/data/processed/processed_ratings_train.txt --test_corpus_path /notebooks/embedding/data/processed/processed_ratings_test.txt --embedding_corpus_path /notebooks/embedding/data/tokenized/corpus_mecab.txt --embedding_path /notebooks/embedding/data/word-embeddings/word2vec/word2vec --output_path /notebooks/embedding/data/word-embeddings/cbow/word2vec --embedding_name word2vec --method cbow --is_weighted True
        # fasttext - original
        python /notebooks/embedding/models/word_utils.py --train_corpus_path /notebooks/embedding/data/processed/processed_ratings_train.txt --test_corpus_path /notebooks/embedding/data/processed/processed_ratings_test.txt --embedding_path /notebooks/embedding/data/word-embeddings/fasttext/fasttext.vec --output_path /notebooks/embedding/data/word-embeddings/cbow/fasttext --embedding_name fasttext --method cbow --is_weighted False
        # fasttext - weighted
        python /notebooks/embedding/models/word_utils.py --train_corpus_path /notebooks/embedding/data/processed/processed_ratings_train.txt --test_corpus_path /notebooks/embedding/data/processed/processed_ratings_test.txt --embedding_corpus_path /notebooks/embedding/data/tokenized/corpus_mecab.txt --embedding_path /notebooks/embedding/data/word-embeddings/fasttext/fasttext.vec --output_path /notebooks/embedding/data/word-embeddings/cbow/fasttext --embedding_name fasttext --method cbow --is_weighted True
        # glove - original
        python /notebooks/embedding/models/word_utils.py --train_corpus_path /notebooks/embedding/data/processed/processed_ratings_train.txt --test_corpus_path /notebooks/embedding/data/processed/processed_ratings_test.txt --embedding_path /notebooks/embedding/data/word-embeddings/glove/glove.txt --output_path /notebooks/embedding/data/word-embeddings/cbow/glove --embedding_name glove --method cbow --is_weighted False
        # glove - weighted
        python /notebooks/embedding/models/word_utils.py --train_corpus_path /notebooks/embedding/data/processed/processed_ratings_train.txt --test_corpus_path /notebooks/embedding/data/processed/processed_ratings_test.txt --embedding_corpus_path /notebooks/embedding/data/tokenized/corpus_mecab.txt --embedding_path /notebooks/embedding/data/word-embeddings/glove/glove.txt --output_path /notebooks/embedding/data/word-embeddings/cbow/glove --embedding_name glove --method cbow --is_weighted True
        # swivel - original
        python /notebooks/embedding/models/word_utils.py --train_corpus_path /notebooks/embedding/data/processed/processed_ratings_train.txt --test_corpus_path /notebooks/embedding/data/processed/processed_ratings_test.txt --embedding_path /notebooks/embedding/data/word-embeddings/swivel/row_embedding.tsv --output_path /notebooks/embedding/data/word-embeddings/cbow/swivel --embedding_name swivel --method cbow --is_weighted False
        # swivel - weighted
        python /notebooks/embedding/models/word_utils.py --train_corpus_path /notebooks/embedding/data/processed/processed_ratings_train.txt --test_corpus_path /notebooks/embedding/data/processed/processed_ratings_test.txt --embedding_corpus_path /notebooks/embedding/data/tokenized/corpus_mecab.txt --embedding_path /notebooks/embedding/data/word-embeddings/swivel/row_embedding.tsv --output_path /notebooks/embedding/data/word-embeddings/cbow/swivel --embedding_name swivel --method cbow --is_weighted True
        ;;
esac