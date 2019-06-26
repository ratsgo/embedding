---
layout: default
title: 데이터 전처리
description: 원시 말뭉치를 내려받아 텍스트 형태로 가공합니다.
---



### 데이터 덤프

아래의 `wget` 명령어를 입력해 필요한 말뭉치를 다운로드합니다.

#### [네이버 영화 말뭉치](https://github.com/e9t/nsmc)

```bash
wget https://github.com/e9t/nsmc/raw/master/ratings.txt -P /notebooks/embedding/data/raw
wget https://github.com/e9t/nsmc/raw/master/ratings_train.txt -P /notebooks/embedding/data/raw
wget https://github.com/e9t/nsmc/raw/master/ratings_test.txt -P /notebooks/embedding/data/raw
```

#### 한국어 위키피디아

```bash
wget https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2 -P /notebooks/embedding/data/raw
```

#### [KorQuAD](https://korquad.github.io)

```bash
wget https://korquad.github.io/dataset/KorQuAD_v1.0_train.json -P /notebooks/embedding/data/raw
wget https://korquad.github.io/dataset/KorQuAD_v1.0_dev.json -P /notebooks/embedding/data/raw
```

#### [유사 문장](https://github.com/songys/Question_pair)

```bash
wget https://github.com/songys/Question_pair/raw/master/kor_pair_train.csv -P /notebooks/embedding/data/raw
wget https://github.com/songys/Question_pair/raw/master/kor_Pair_test.csv -P /notebooks/embedding/data/raw
```

#### [ratsgo blog](http://ratsgo.github.io)

```bash
function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}
          
gdrive_download 1Few7-Mh3JypQN3rjnuXD8yAXrkxUwmjS /notebooks/embedding/data/processed/processed_blog.txt
```

  
### 데이터를 문장 단위 텍스트 파일로 저장하기

`/notebooks/embedding` 위치에서 다음을 실행하면 각기 다른 형식의 데이터를 한 라인이 한 문서인 형태의 텍스트 파일로 저장합니다. 이 단계에서는 별도로 토크나이즈를 하진 않습니다.

#### 네이버 영화 말뭉치 전처리

json, text 형태의 영화 리뷰를 처리합니다.

```bash
python preprocess/dump.py --preprocess_mode nsmc \
	--input_path /notebooks/embedding/data/raw/ratings.txt \
	--output_path /notebooks/embedding/data/processed/processed_ratings.txt \
	--with_label False
python preprocess/dump.py --preprocess_mode nsmc \
	--input_path /notebooks/embedding/data/raw/ratings_train.txt \
	--output_path /notebooks/embedding/data/processed/processed_ratings_train.txt \
	--with_label True
python preprocess/dump.py --preprocess_mode nsmc \
	--input_path /notebooks/embedding/data/raw/ratings_test.txt \
	--output_path /notebooks/embedding/data/processed/processed_ratings_test.txt \
	--with_label True
```

#### 한국어 위키피디아 말뭉치 전처리

위키피디아 원문에서 이메일, URL, 여러 형태의 공백 등 불필요 문자를 제거하고 숫자 사이에 공백을 추가하는 등의 전처리를 시행합니다. 

```bash
python preprocess/dump.py --preprocess_mode wiki \
	--input_path /notebooks/embedding/data/raw/kowiki-latest-pages-articles.xml.bz2 \
	--output_path /notebooks/embedding/data/processed/processed_wiki_ko.txt
```

#### KorQuad 데이터 전처리

json 내 context를 문서 하나로 취급합니다. question, anwers은 두 쌍을 공백으로 묶어서 문서 하나로 취급합니다.

```bash
python preprocess/dump.py --preprocess_mode korsquad \
	--input_path /notebooks/embedding/data/raw/KorQuAD_v1.0_train.json \
	--output_path /notebooks/embedding/data/processed/processed_korsquad_train.txt
python preprocess/dump.py --preprocess_mode korsquad \
	--input_path /notebooks/embedding/data/raw/KorQuAD_v1.0_dev.json \
	--output_path data/processed/processed_korsquad_dev.txt
```

다음 한 줄이면 위 명령을 수행하지 않고도 이미 처리된 파일들을 한꺼번에 내려받을 수 있습니다. 

```bash
gdrive_download 1hscU5_f_1vXfbhHabNpqfnp8DU2ZWmcT /notebooks/embedding/data/processed.zip
```
