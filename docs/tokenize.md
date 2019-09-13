---
layout: default
title: 형태소 분석
description: 말뭉치를 형태소 분석하는 방법을 안내합니다.
---



## 형태소 분석

이 페이지에서는 말뭉치를 형태소 분석하는 방법을 안내합니다.
`/notebooks/embedding` 위치에서 다음을 실행하면 각 데이터를 형태소 분석할 수 있습니다. 
입력 파일은 한 라인이 한 문서인 형태여야 합니다. 



### 1. supervised tokenizer

은전한닢 등 5개의 한국어 형태소 분석기를 지원합니다. 사용법은 다음과 같습니다.

| 형태소 분석기 | 명령                                                         |
| ------------- | ------------------------------------------------------------ |
| 은전한닢      | python preprocess/supervised_nlputils.py --tokenizer `mecab` --input_path `input_file_path` --output_path `output_file_path` |
| 코모란        | python preprocess/supervised_nlputils.py --tokenizer `komoran` --input_path `input_file_path` --output_path `output_file_path` |
| Okt           | python preprocess/supervised_nlputils.py --tokenizer `okt` --input_path `input_file_path` --output_path `output_file_path` |
| 한나눔        | python preprocess/supervised_nlputils.py --tokenizer `hannanum` --input_path `input_file_path` --output_path `output_file_path` |
| Khaiii        | python preprocess/supervised_nlputils.py --tokenizer `khaiii` --input_path `input_file_path` --output_path `output_file_path` |



### 2. unsupervised tokenizer

soynlp와 구글 SentencePiece 두 가지 분석기를 지원합니다. supervised tokenizer들과 달리 말뭉치의 통계량을 확인한 뒤 토크나이즈를 하기 때문에 토크나이즈 적용 전 모델 학습이 필요합니다. 말뭉치 데이터가 필요하다면 [이곳](https://ratsgo.github.io/embedding/downloaddata.html)에 방문하셔서 필요한 데이터를 내려 받으세요.



#### 2.1. soynlp

[soynlp 형태소 분석 모델](https://github.com/lovit/soynlp)을 학습합니다. `input_path`는 학습데이터의 위치, `model_path`는 학습된 모델을 저장할 위치를 가리킵니다.

```bash
# train
python preprocess/unsupervised_nlputils.py --preprocess_mode compute_soy_word_score \
	--input_path /notebooks/embedding/data/processed/corrected_ratings_corpus.txt \
	--model_path /notebooks/embedding/data/processed/soyword.model
```

형태소를 분석합니다.  `input_path`는 형태소 분석할 말뭉치의 위치, `model_path`는 학습 완료된 형태소 분석 모델의 저장 위치, `output_path` 는 형태소 분석이 완료된 말뭉치가 저장될 위치를 의미합니다.

```bash
# tokenize
python preprocess/unsupervised_nlputils.py --preprocess_mode soy_tokenize \
	--input_path /notebooks/embedding/data/processed/corrected_ratings_corpus.txt \
	--model_path /notebooks/embedding/data/processed/soyword.model \
	--output_path /notebooks/embedding/data/tokenized/ratings_soynlp.txt
```



#### 2.2. sentencepiece

[구글 sentencepiece 패키지](https://github.com/google/sentencepiece)를 활용해 `Byte Pair Encoding` 기법으로 단어 사전(vocabulary)을 만들고, 이를 BERT 모델에 맞도록 후처리합니다. `input_path`는 BPE 학습데이터 위치, `vocab_path`는 이 모듈의 최종 처리 결과인 BERT 모델에 쓸 단어 사전이 저장될 위치입니다.

```bash
python preprocess/unsupervised_nlputils.py --preprocess_mode make_bert_vocab \
	--input_path /notebooks/embedding/data/processed/corrected_ratings_corpus.txt \
	--vocab_path /notebooks/embedding/data/processed/bert.vocab
```

위에서 만든 사전으로 `input_path`에 있는 말뭉치를 토크나이즈한 후 `output_path`에 저장합니다.

```bash
python preprocess/unsupervised_nlputils.py --preprocess_mode bert_tokenize \
	--vocab_path /notebooks/embedding/data/processed/bert.vocab \
	--input_path /notebooks/embedding/data/processed/corrected_ratings_corpus.txt \
	--output_path /notebooks/embedding/data/tokenized/ratings_sentpiece.txt
```



### 3. 띄어쓰기 교정

`supervised tokenizer`이든, `unsupervised tokenizer`이든 띄어쓰기가 잘 되어 있을 수록 형태소 분석 품질이 좋아집니다. [soynlp](https://github.com/lovit/soynlp)의 띄어쓰기 교정 모델을 학습하는 코드는 다음과 같습니다.

```bash
python preprocess/unsupervised_nlputils.py --preprocess_mode train_space \
	--input_path /notebooks/embedding/data/processed/processed_ratings.txt \
	--model_path /notebooks/embedding/data/trained-models/space-correct.model
```

학습된 띄어쓰기 모델로 말뭉치를 교정하는 과정은 다음과 같습니다.

```bash
python preprocess/unsupervised_nlputils.py --preprocess_mode apply_space_correct \
	--input_path /notebooks/embedding/data/processed/processed_ratings.txt \
	--model_path /notebooks/embedding/data/trained-models/space-correct.model \
	--output_path /notebooks/embedding/data/processed/corrected_ratings_corpus.txt \
	--with_label False
```



### 4. 형태소 분석 완료된 데이터 다운로드

형태소 분석에 시간을 투자하고 싶지 않은 분들은 아래를 실행하면 형태소 분석이 모두 완료된 데이터들을 한꺼번에 다운로드할 수 있습니다. 이밖에 다른 데이터를 내려받고 싶다면 [이 글](https://ratsgo.github.io/embedding/downloaddata.html)을 참고하세요.

```bash
bash preprocess.sh dump-tokenized
```

