#!/usr/bin/env bash

COMMAND=$1

case $COMMAND in
    lsa-tfidf)
        echo "latent semantic analysis with tf-idf matrix..."
        python models/sent_utils.py latent_semantic_analysis data/tmp.txt data/lsa-tfidf.vecs
        ;;
    doc2vec)
        echo "doc2vec..."
        python models/sent_utils.py doc2vec data/tmp.txt data/doc2vec.vecs
        ;;
    lda)
        echo "latent_dirichlet_allocation..."
        python models/sent_utils.py latent_dirichlet_allocation data/tmp.txt data/lda
esac