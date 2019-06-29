---
layout: default
title: 데이터 다운로드
description: 말뭉치나 임베딩을 다운로드하는 방법을 안내합니다. 
---



## 원본 데이터

한국어 공개 말뭉치를 내려받습니다. 그 목록은 다음과 같습니다.

- 한국어 위키백과 : [다운로드](https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2)
- [KorQuAD](https://korquad.github.io) 학습 데이터셋 : [다운로드](https://korquad.github.io/dataset/KorQuAD_v1.0_train.json)
- [KorQuAD](https://korquad.github.io) 데브 데이터셋 : [다운로드](https://korquad.github.io/dataset/KorQuAD_v1.0_dev.json)
- [네이버 영화 리뷰](https://github.com/e9t/nsmc) 학습 데이터셋 : [다운로드](https://github.com/e9t/nsmc/raw/master/ratings_train.txt)
- [네이버 영화 리뷰](https://github.com/e9t/nsmc) 테스트 데이터셋 : [다운로드](https://github.com/e9t/nsmc/raw/master/ratings_test.txt)





## 전처리 데이터

[이곳](https://drive.google.com/open?id=1kUecR7xO7bsHFmUI6AExtY5u2XXlObOG)을 클릭하시면 전처리가 완료된 데이터를 내려받을 수 있습니다. 순수 텍스트 파일이며 1개 라인에 1개 문서가 기록돼 있습니다. 그 목록은 다음과 같습니다.

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

[이곳](https://drive.google.com/open?id=1Ybp_DmzNEpsBrUKZ1-NoPDzCMO39f-fx)을 클릭하시면 형태소 분석이 완료된 데이터를 내려받을 수 있습니다. 순수 텍스트 파일이며 1개 라인에 1개 문서가 기록돼 있습니다. 그 목록은 다음과 같습니다.

| 파일명                | 데이터                                       | 형태소 분석기                                    |
| --------------------- | -------------------------------------------- | ------------------------------------------------ |
| corpus_mecab_jamo.txt | 한국어 위키백과, 네이버 영화 말뭉치, KorQuAD | 은전한닢 + 자소 분해                             |
| korquad_mecab.txt     | KorQuAD                                      | 은전한닢                                         |
| ratings_hannanum.txt  | 네이버 영화 말뭉치                           | 한나눔                                           |
| ratings_khaiii.txt    | 네이버 영화 말뭉치                           | Khaiii                                           |
| ratings_komoran.txt   | 네이버 영화 말뭉치                           | 코모란                                           |
| ratings_mecab.txt     | 네이버 영화 말뭉치                           | 은전한닢                                         |
| ratings_okt.txt       | 네이버 영화 말뭉치                           | Okt                                              |
| ratings_soynlp.txt    | 네이버 영화 말뭉치                           | [soynlp 패키지](https://github.com/lovit/soynlp) |
| wiki_ko_mecab.txt     | 한국어 위키백과                              | 은전한닢                                         |

혹은 우분투 쉘(도커 컨테이너, [개발환경 설정 참고](https://ratsgo.github.io/embedding/environment.html))의 `/notebooks/embedding` 디렉토리에서 아래 스크립트를 실행하면 `/notebooks/embedding/data/tokenized` 디렉토리에 모든 파일을 한꺼번에 내려받습니다.

```bash
git pull origin master
bash preprocess.sh dump-tokenized
```





## 단어 임베딩

[이곳](https://drive.google.com/open?id=1gpOaOl0BcUvYpgoOA2JpZY2z-BUhuBLX)을 클릭하시면 학습이 완료된 단어 수준 임베딩을 내려받을 수 있습니다. `FastText-Jamo`를 제외한 모든 임베딩은 한국어 위키백과, KorQuAD, 네이버 영화 말뭉치를 은전한닢(mecab)으로 형태소 분석한 말뭉치로 학습됐습니다. `FastText-Jamo`의 학습데이터는 `corpus_mecab_jamo.txt`입니다.



- FastText
- FastText-Jamo
- GloVe
- Swivel
- Word2Vec



혹은 우분투 쉘(도커 컨테이너, [개발환경 설정 참고](https://ratsgo.github.io/embedding/environment.html))의 `/notebooks/embedding` 디렉토리에서 아래 스크립트를 실행하면 `/notebooks/embedding/data/word-embeddings` 디렉토리에 모든 파일을 한꺼번에 내려받습니다.

```bash
git pull origin master
bash preprocess.sh dump-word-embeddings
```





## 문장 임베딩

[이곳](https://drive.google.com/open?id=1jL3Q5H1vwATewHrx0PJgJ8YoUCtEkaGW)을 클릭하시면 학습이 완료된 문장 수준 임베딩을 내려받을 수 있습니다. 그 목록은 다음과 같습니다.

| 파일명 | 내용                                                         |
| ------ | ------------------------------------------------------------ |
| bert   | (1) 구글이 공개한 다국어 BERT 임베딩 (multi_cased_L-12_H-768_A-12)<br />(2) (1)을 네이버 영화 말뭉치로 파인튜닝한 BERT 모델 |
| elmo   | (1) 한국어 위키백과, KorQuAD, 네이버 영화 말뭉치를 은전한닢으로 형태소 분석한 뒤 학습<br />(2) (1)을 네이버 영화 말뭉치로 파인튜닝한 ELMo 모델 |

혹은 우분투 쉘(도커 컨테이너, [개발환경 설정 참고](https://ratsgo.github.io/embedding/environment.html))의 `/notebooks/embedding` 디렉토리에서 아래 스크립트를 실행하면 `/notebooks/embedding/data/sentence-embeddings` 디렉토리에 모든 파일을 한꺼번에 내려받습니다.

```bash
git pull origin master
bash preprocess.sh dump-sentence-embeddings
```

