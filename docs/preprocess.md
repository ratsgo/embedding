---
layout: default
title: 데이터 전처리
description: 원시 말뭉치를 내려받아 텍스트 형태로 가공합니다.
---



## 데이터 전처리

이 페이지에서는 원시 말뭉치를 내려받아 텍스트 형태로 가공하는 방법을 안내합니다.



### 1. 데이터 덤프

아래의 명령어를 입력해 필요한 말뭉치를 다운로드합니다.



#### 1.1. [네이버 영화 말뭉치](https://github.com/e9t/nsmc)

```bash
bash preprocess.sh dump-raw-nsmc
```



#### 1.2. 한국어 위키피디아

```bash
bash preprocess.sh dump-raw-wiki
```



#### 1.3. [KorQuAD](https://korquad.github.io)

```bash
bash preprocess.sh dump-raw-korquad
```



#### 1.4. [유사 문장](https://github.com/songys/Question_pair)

```bash
wget https://github.com/songys/Question_pair/raw/master/kor_pair_train.csv -P /notebooks/embedding/data/raw
wget https://github.com/songys/Question_pair/raw/master/kor_Pair_test.csv -P /notebooks/embedding/data/raw
```



#### 1.5. [ratsgo blog](http://ratsgo.github.io)

```bash
function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}
          
gdrive_download 1Few7-Mh3JypQN3rjnuXD8yAXrkxUwmjS /notebooks/embedding/data/processed/processed_blog.txt
```



### 2. 데이터를 문장 단위 텍스트 파일로 저장하기

`/notebooks/embedding` 위치에서 다음을 실행하면 각기 다른 형식의 데이터를 한 라인이 한 문서인 형태의 텍스트 파일로 저장합니다. 이 단계에서는 별도로 토크나이즈를 하진 않습니다.



#### 2.1. 네이버 영화 말뭉치 전처리

json, text 형태의 영화 리뷰를 처리합니다.

```bash
bash preprocess.sh process-nsmc
```



#### 2.2. 한국어 위키피디아 말뭉치 전처리

위키피디아 원문에서 이메일, URL, 여러 형태의 공백 등 불필요 문자를 제거하고 숫자 사이에 공백을 추가하는 등의 전처리를 시행합니다. 

```bash
bash preprocess.sh process-wiki
```



#### 2.3. KorQuAD 데이터 전처리

json 내 context를 문서 하나로 취급합니다. question, anwers은 두 쌍을 공백으로 묶어서 문서 하나로 취급합니다.

```bash
bash preprocess.sh process-korquad
```



### 3. 전처리 완료된 데이터 다운로드

전처리에 시간을 투자하고 싶지 않은 분들은 아래를 실행하면 전처리가 모두 완료된 데이터들을 한꺼번에 다운로드할 수 있습니다. 이밖에 다른 데이터를 내려받고 싶다면  [이 글](https://ratsgo.github.io/embedding/downloaddata.html)을 참고하세요. 

```bash
bash preprocess.sh dump-processed
```