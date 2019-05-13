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

# khai
cd ~/works
# brew install cmake
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

# sentence piece
cd ~/works
git clone https://github.com/google/sentencepiece.git
cd sentencepiece
mkdir build
cmake ..
make -j $(nproc)
sudo make install
# sudo ldconfig -v # except macOS
sudo update_dyld_shared_cache # macOS

# glove
cd ~/works
git clone http://github.com/stanfordnlp/glove
cd glove && make

# fasttext
cd ~/works
git clone https://github.com/facebookresearch/fastText.git
cd fastText && make