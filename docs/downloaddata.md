---
layout: default
title: 데이터 다운로드
description: 말뭉치나 임베딩을 다운로드하는 방법을 안내합니다. 
---



## 원본 데이터

한국어 공개 말뭉치를 내려받습니다. 그 목록은 다음과 같습니다.

| 말뭉치                                                       | 다운로드 링크                                                |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 한국어 위키백과                                              | https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2 |
| [KorQuAD](https://korquad.github.io) 학습데이터셋            | https://korquad.github.io/dataset/KorQuAD_v1.0_train.json    |
| [KorQuAD](https://korquad.github.io) 데브데이터셋            | https://korquad.github.io/dataset/KorQuAD_v1.0_dev.json      |
| [네이버 영화 리뷰](https://github.com/e9t/nsmc) 학습데이터셋 | https://github.com/e9t/nsmc/raw/master/ratings_train.txt     |
| [네이버 영화 리뷰](https://github.com/e9t/nsmc) 학습데이터셋 | https://github.com/e9t/nsmc/raw/master/ratings_test.txt      |





## 전처리 데이터

[이곳](https://drive.google.com/open?id=1oO5v6YqNlKTq0vWfjME3SiLXAYCMAmkc)을 클릭하시면 전처리가 완료된 데이터를 내려받을 수 있습니다. 순수 텍스트 파일이며 1개 라인에 1개 문서가 기록돼 있습니다. 그 목록은 다음과 같습니다.

| 파일명                       | 내용                                                         |
| ---------------------------  | ------------------------------------------------------------ |
| processed_wiki_ko.txt        | 한국어 위키백과                                              |
| processed_korquad.txt        | KorQuAD 학습/데브셋                                          |
| processed_ratings.txt        | 네이버 영화 말뭉치 학습/테스트셋 (극성 레이블 없음)               |
| processed_ratings_train.txt  | 네이버 영화 말뭉치 학습셋 (극성 레이블 있음)                      |
| processed_ratings_test.txt   | 네이버 영화 말뭉치 테스트셋 (극성 레이블 있음)                    |
| processed_review_movieid.txt | 네이버 영화 말뭉치 전체 데이터셋 (영화 ID 포함)                  |
| space-correct.model          | 네이버 영화 말뭉치로 학습한 [띄어쓰기 교정(soynlp)](https://github.com/lovit/soynlp) 모델 |
| corrected_ratings_train.txt  | [띄어쓰기 교정(soynlp)](https://github.com/lovit/soynlp)한 네이버 영화 말뭉치 학습셋 (레이블 있음) |
| corrected_ratings_test.txt   | [띄어쓰기 교정(soynlp)](https://github.com/lovit/soynlp)한 네이버 영화 말뭉치 테스트셋 (레이블 없음) |
| soyword.model                | 네이버 영화 말뭉치로 학습한 [soynlp](https://github.com/lovit/soynlp) 형태소 분석 모델 |

혹은 우분투 쉘(도커 컨테이너, [개발환경 설정 참고](https://ratsgo.github.io/embedding/environment.html))의 `/notebooks/embedding` 디렉토리에서 아래 스크립트를 실행하면 `/notebooks/embedding/data/processed` 디렉토리에 모든 파일을 한꺼번에 내려받습니다.

```bash
git pull origin master
bash preprocess.sh dump-processed
```





## 형태소 분석 데이터

[이곳](https://drive.google.com/open?id=1QEdjvT0Jpqmz9F57ATmjy3i016kdcURq)을 클릭하시면 형태소 분석이 완료된 데이터를 내려받을 수 있습니다. 순수 텍스트 파일이며 1개 라인에 1개 문서가 기록돼 있습니다. 그 목록은 다음과 같습니다.

| 파일명                | 내용                                                         |
| --------------------- | ------------------------------------------------------------ |
| corpus_mecab_jamo.txt | 한국어 위키백과, 네이버 영화 말뭉치, KorQuAD를 합치고 은전한닢(mecab)으로 형태소 분석을 한 뒤 자소로 분해한 데이터셋 |
| korquad_mecab.txt     | KorQuAD를 은전한닢으로 형태소 분석한 데이터셋                |
| ratings_hannanum.txt  | 네이버 영화 말뭉치를 한나눔으로 형태소 분석한 데이터셋       |
| ratings_khaiii.txt    | 네이버 영화 말뭉치를 Khaiii로 형태소 분석한 데이터셋         |
| ratings_komoran.txt   | 네이버 영화 말뭉치를 코모란으로 형태소 분석한 데이터셋       |
| ratings_mecab.txt     | 네이버 영화 말뭉치를 은전한닢으로 형태소 분석한 데이터셋     |
| ratings_okt.txt       | 네이버 영화 말뭉치를 Okt로 형태소 분석한 데이터셋            |
| ratings_sentpiece.txt | 네이버 영화 말뭉치를 [구글 sentencepiece 패키지](https://github.com/google/sentencepiece)로 형태소 분석한 데이터셋 |
| ratings_soynlp.txt    | [soynlp 패키지](https://github.com/lovit/soynlp)로 형태소 분석한 데이터셋 |
| wiki_ko_mecab.txt     | 한국어 위키백과를 은전한닢으로 형태소 분석한 데이터셋        |

혹은 우분투 쉘(도커 컨테이너, [개발환경 설정 참고](https://ratsgo.github.io/embedding/environment.html))의 `/notebooks/embedding` 디렉토리에서 아래 스크립트를 실행하면 `/notebooks/embedding/data/tokenized` 디렉토리에 모든 파일을 한꺼번에 내려받습니다.

```bash
git pull origin master
bash preprocess.sh dump-tokenized
```





## 단어 임베딩

[이곳](https://drive.google.com/open?id=1gpOaOl0BcUvYpgoOA2JpZY2z-BUhuBLX)을 클릭하시면 학습이 완료된 단어 수준 임베딩을 내려받을 수 있습니다. 그 목록은 다음과 같습니다.

| 디렉토리명    | 내용                                                         |
| ------------- | ------------------------------------------------------------ |
| fasttext      | 한국어 위키백과, KorQuAD, 네이버 영화 말뭉치를 은전한닢으로 형태소 분석한 뒤 학습한 FastText 임베딩 |
| fasttext-jamo | 한국어 위키백과, KorQuAD, 네이버 영화 말뭉치를 은전한닢으로 형태소 분석한 뒤 자소로 분해한 다음 FastText 임베딩 |
| glove         | 한국어 위키백과, KorQuAD, 네이버 영화 말뭉치를 은전한닢으로 형태소 분석한 뒤 GloVe 임베딩 |
| swivel        | 한국어 위키백과, KorQuAD, 네이버 영화 말뭉치를 은전한닢으로 형태소 분석한 뒤 Swivel 임베딩 |
| word2vec      | 한국어 위키백과, KorQuAD, 네이버 영화 말뭉치를 은전한닢으로 형태소 분석한 뒤 Word2Vec 임베딩 |

혹은 우분투 쉘(도커 컨테이너, [개발환경 설정 참고](https://ratsgo.github.io/embedding/environment.html))의 `/notebooks/embedding` 디렉토리에서 아래 스크립트를 실행하면 `/notebooks/embedding/data/word-embeddings` 디렉토리에 모든 파일을 한꺼번에 내려받습니다.

```bash
git pull origin master
bash preprocess.sh dump-word-embeddings
```





## 문장 임베딩

[이곳](https://drive.google.com/open?id=1y_58tgW4S9ujOrwUs9oUtLMM6MfGhctr)을 클릭하시면 학습이 완료된 문장 수준 임베딩을 내려받을 수 있습니다. 그 목록은 다음과 같습니다.

| 파일명  | 내용                                                         |
| ------- | ------------------------------------------------------------ |
| bert    | (1) 구글이 공개한 다국어 BERT 임베딩 (multi_cased_L-12_H-768_A-12)<br />(2) (1)을 네이버 영화 말뭉치로 파인튜닝한 BERT 모델 |
| elmo    | (1) 한국어 위키백과, KorQuAD, 네이버 영화 말뭉치를 은전한닢으로 형태소 분석한 뒤 학습한 ELMo 임베딩<br />(2) (1)을 네이버 영화 말뭉치로 파인튜닝한 ELMo 모델 |

혹은 우분투 쉘(도커 컨테이너, [개발환경 설정 참고](https://ratsgo.github.io/embedding/environment.html))의 `/notebooks/embedding` 디렉토리에서 아래 스크립트를 실행하면 `/notebooks/embedding/data/sentence-embeddings` 디렉토리에 모든 파일을 한꺼번에 내려받습니다.

```bash
git pull origin master
bash preprocess.sh dump-sentence-embeddings
```

