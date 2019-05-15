#!/usr/bin/env bash

# set paths
./path.sh

# install brew packages
brew install cmake
brew install protobuf

# make data directory
mkdir $EMBEDDING_MAIN_PATH/data

# tensorflow
pip install tensorflow

# gensim
pip install gensim

# soynlp
pip install soynlp

# naver movie corpus
cd $PARENT_PATH
git clone https://github.com/e9t/nsmc.git

# konlpy
pip install konlpy
bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)

# khai
cd $PARENT_PATH
git clone https://github.com/kakao/khaiii.git
cd khaiii
mkdir build
cd build
cmake ..
make all
make resource
ctest
make package_python
pip install  .

# sentence piece
cd $PARENT_PATH
git clone https://github.com/google/sentencepiece.git
cd sentencepiece
mkdir build
cmake ..
make -j $(nproc)
sudo make install
sudo update_dyld_shared_cache # macOS

# pytorch bert (for BERT tokenizer)
pip install pytorch_pretrained_bert

# glove
cd $PARENT_PATH
git clone http://github.com/stanfordnlp/glove
cd glove && make

# fasttext
cd $PARENT_PATH
git clone https://github.com/facebookresearch/fastText.git
cd fastText && make

# swivel
cd $PARENT_PATH
git clone https://github.com/tensorflow/models.git
cd models/research/swivel
make -f fastprep.mk

# elmo
cd $PARENT_PATH
git clone https://github.com/allenai/bilm-tf.git

# bert
cd $PARENT_PATH
git clone https://github.com/google-research/bert.git

# workstation
python3.6 -m venv tf120
pip install tensorflow-gpu==1.12
pip install h5py