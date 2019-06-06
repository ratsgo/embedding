#!/usr/bin/env bash

#!/usr/bin/env bash

COMMAND=$1
export LC_CTYPE=C.UTF-8
case $COMMAND in
    merge)
        echo "merge datasets.."
        cat /notebooks/embedding/data/tokenized/wiki_ko_mecab.txt /notebooks/embedding/data/tokenized/ratings_mecab.txt /notebooks/embedding/data/tokenized/korquad_mecab.txt > /notebooks/embedding/data/tokenized/corpus_mecab.txt
        cat /notebooks/embedding/data/tokenized/ratings_mecab.txt /notebooks/embedding/data/tokenized/korquad_mecab.txt > /notebooks/embedding/data/tokenized/for-lsa-mecab.txt
        ;;
    lsa)
        echo "latent semantic analysis..."
        mkdir -p /notebooks/embedding/data/word-embeddings/lsa
        python models/word_utils.py --method latent_semantic_analysis \
            --input_path /notebooks/embedding/data/tokenized/for-lsa-mecab.txt \
            --output_path /notebooks/embedding/data/word-embeddings/lsa/lsa
        ;;
    word2vec)
        echo "word2vec word embedding..."
        mkdir -p /notebooks/embedding/data/word-embeddings/word2vec
        python models/word_utils.py --method train_word2vec \
            --input_path /notebooks/embedding/data/tokenized/corpus_mecab.txt \
            --output_path /notebooks/embedding/data/word-embeddings/word2vec/word2vec
        python models/word_utils.py --method train_word2vec \
            --input_path /notebooks/embedding/data/tokenized/for-lsa-mecab.txt \
            --output_path /notebooks/embedding/data/word-embeddings/word2vec/word2vec-lsa
        ;;
    glove)
        echo "glove word embedding..."
        # 각 파일 역할
        # https://github.com/stanfordnlp/GloVe/tree/master/src
        mkdir -p /notebooks/embedding/data/word-embeddings/glove
        /notebooks/embedding/models/glove/build/vocab_count -min-count 5 -verbose 2 < /notebooks/embedding/data/tokenized/corpus_mecab.txt > /notebooks/embedding/data/word-embeddings/glove/glove.vocab
        /notebooks/embedding/models/glove/build/cooccur -memory 10.0 -vocab-file /notebooks/embedding/data/word-embeddings/glove/glove.vocab -verbose 2 -window-size 15 < /notebooks/embedding/data/tokenized/corpus_mecab.txt > /notebooks/embedding/data/word-embeddings/glove/glove.cooc
        /notebooks/embedding/models/glove/build/shuffle -memory 10.0 -verbose 2 < /notebooks/embedding/data/word-embeddings/glove/glove.cooc > /notebooks/embedding/data/word-embeddings/glove/glove.shuf
        /notebooks/embedding/models/glove/build/glove -save-file /notebooks/embedding/data/word-embeddings/glove/glove.vecs -threads 4 -input-file /notebooks/embedding/data/word-embeddings/glove/glove.shuf -x-max 10 -iter 15 -vector-size 100 -binary 2 -vocab-file /notebooks/embedding/data/word-embeddings/glove/glove.vocab -verbose 2
        ;;
    fasttext)
        echo "fasttext word embedding..."
        mkdir -p /notebooks/embedding/data/word-embeddings/fasttext
        /notebooks/embedding/models/fastText/fasttext skipgram -input /notebooks/embedding/data/tokenized/corpus_mecab.txt -output /notebooks/embedding/data/word-embeddings/fasttext/fasttext
        ;;
    fasttext-jamo)
        echo "fasttext jamo embedding..."
        mkdir -p /notebooks/embedding/data/word-embeddings/fasttext-jamo
        /notebooks/embedding/models/fastText/fasttext skipgram -input /notebooks/embedding/data/tokenized/corpus_mecab_jamo.txt -output /notebooks/embedding/data/word-embeddings/fasttext-jamo/fasttext-jamo
        ;;
    swivel)
        echo "swivel word embedding..."
        mkdir -p /notebooks/embedding/data/word-embeddings/swivel
        /notebooks/embedding/models/swivel/fastprep --input /notebooks/embedding/data/tokenized/corpus_mecab.txt --output_dir /notebooks/embedding/data/word-embeddings/swivel/swivel.data
        python /notebooks/embedding/models/swivel/swivel.py --input_base_path /notebooks/embedding/data/word-embeddings/swivel/swivel.data --output_base_path /notebooks/embedding/data/word-embeddings/swivel --dim 100
        ;;
esac