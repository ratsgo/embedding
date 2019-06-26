---
layout: default
title: 형태소 분석
description: 말뭉치를 형태소 분석하는 방법을 안내합니다.
---


# 형태소 분석

- 이 페이지에서는 말뭉치를 형태소 분석하는 방법을 안내합니다.
- `/notebooks/embedding` 위치에서 다음을 실행하면 각 데이터를 형태소 분석할 수 있습니다. 
- 입력 파일은 한 라인이 한 문서인 형태여야 합니다. 


## supervised tokenizer

- 은전한닢 등 5개의 한국어 형태소 분석기를 지원합니다. 사용법은 다음과 같습니다.

| 형태소 분석기 | 명령                                                         |
| ------------- | ------------------------------------------------------------ |
| 은전한닢      | python preprocess/supervised_nlputils.py --tokenizer mecab --input_path input_file_path --output_path output_file_path |
| 코모란        | python preprocess/supervised_nlputils.py --tokenizer komoran --input_path input_file_path --output_path output_file_path |
| Okt           | python preprocess/supervised_nlputils.py --tokenizer okt --input_path input_file_path --output_path output_file_path |
| 한나눔        | python preprocess/supervised_nlputils.py --tokenizer hannanum --input_path input_file_path --output_path output_file_path |
| Khaiii        | python preprocess/supervised_nlputils.py --tokenizer khaiii --input_path input_file_path --output_path output_file_path |


## unsupervised tokenizer

- soynlp와 구글 SentencePiece 두 가지 분석기를 지원합니다. 
- supervised tokenizer들과 달리 말뭉치의 통계량을 확인한 뒤 토크나이즈를 하기 때문에 토크나이즈 적용 전 모델 학습이 필요합니다.


### soynlp

- 사용법은 다음과 같습니다.

```bash
# train
python preprocess/unsupervised_nlputils.py --preprocess_mode compute_soy_word_score \
	--input_path /notebooks/embedding/data/processed/corrected_ratings_corpus.txt \
	--model_path /notebooks/embedding/data/trained-models/soyword.model
# tokenize
python preprocess/unsupervised_nlputils.py --preprocess_mode soy_tokenize \
	--input_path /notebooks/embedding/data/processed/corrected_ratings_corpus.txt \
	--model_path /notebooks/embedding/data/trained-models/soyword.model \
	--output_path /notebooks/embedding/data/tokenized/ratings_soynlp.txt
```

### sentencepiece

- 사용법은 다음과 같습니다.

```bash
# train
cd /notebooks/embedding/data/trained-models
spm_train --input=/notebooks/embedding/data/processed/corrected_ratings_corpus.txt --model_prefix=sentpiece --vocab_size=50000
cd /notebooks/embedding
python preprocess/unsupervised_nlputils.py --preprocess_mode process_sp_vocab \
	--input_path /notebooks/embedding/data/trained-models/sentpiece.vocab \
	--vocab_path /notebooks/embedding/data/trained-models/processed_sentpiece.vocab
# tokenize
python preprocess/unsupervised_nlputils.py --preprocess_mode sentencepiece_tokenize \
	--vocab_path /notebooks/embedding/data/trained-models/processed_sentpiece.vocab \
	--input_path /notebooks/embedding/data/processed/corrected_ratings_corpus.txt \
	--output_path /notebooks/embedding/data/tokenized/ratings_sentpiece.txt
```

## 띄어쓰기 교정

- `supervised tokenizer`이든, `unsupervised tokenizer`이든 띄어쓰기가 잘 되어 있을 수록 형태소 분석 품질이 좋아집니다. 
- soynlp의 띄어쓰기 교정 모델을 학습하고 말뭉치에 모델을 적용하는 과정은 다음과 같습니다.

```bash
# train
python preprocess/unsupervised_nlputils.py --preprocess_mode train_space \
	--input_path /notebooks/embedding/data/processed/processed_ratings.txt \
	--model_path /notebooks/embedding/data/trained-models/space-correct.model
# correct
python preprocess/unsupervised_nlputils.py --preprocess_mode apply_space_correct \
	--input_path /notebooks/embedding/data/processed/processed_ratings.txt \
	--model_path /notebooks/embedding/data/trained-models/space-correct.model \
	--output_path /notebooks/embedding/data/processed/corrected_ratings_corpus.txt \
	--with_label False
```