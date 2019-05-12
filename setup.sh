#!/usr/bin/env bash

# make data directory
mkdir data

# tensorflow
pip install tensorflow

# gensim
pip install gensim

# soynlpy
pip install soynlp

# konlpy
pip install konlpy
bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)

# glove
pip install glove-python

# khai
cd ~/works
brew install cmake
git clone https://github.com/kakao/khaiii.git
cd khaiii
mkdir build
cd build
cmake ..
make all
make resource
# ./bin/khaiii --rsc-dir=./share/khaiii # test
ctest
make package_python
pip install  .

# fasttext
