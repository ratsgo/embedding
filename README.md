# embedding
자연언어처리의 근간이 되는 각종 임베딩 기법들에 관련한 튜토리얼입니다. 한국어 처리를 염두에 두고 작성됐습니다.



### 요구사항

- docker 18.09.1
- (GPU 환경) Nvidia CUDA 9.0 이상



### 도커 실행방법

- nvidia GPU가 있을 때
```bash
docker build -t embedding -f docker/Dockerfile-CPU .
docker run -it --rm embedding-cpu bash
```

- nvidia GPU가 없을 때

```bash
docker build -t embedding -f docker/Dockerfile-GPU .
docker run -it --rm --runtime=nvidia embedding-gpu bash
```



### 데이터 덤프

- 아래의 `wget` 명령어를 입력해 필요한 말뭉치를 다운로드합니다.

- 네이버 영화 말뭉치

```bash
wget https://github.com/e9t/nsmc/raw/master/ratings.txt -P /notebooks/embedding/data
wget https://github.com/e9t/nsmc/raw/master/ratings_train.txt -P /notebooks/embedding/data
wget https://github.com/e9t/nsmc/raw/master/ratings_test.txt -P /notebooks/embedding/data
```

- 한국어 위키피디아

```bash
wget https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2 -P /notebooks/embedding/data
```

- KorSquad

```bash
wget https://korquad.github.io/dataset/KorQuAD_v1.0_train.json -P /notebooks/embedding/data
wget https://korquad.github.io/dataset/KorQuAD_v1.0_dev.json -P /notebooks/embedding/data
```

- 유사 문장

```bash
wget https://github.com/songys/Question_pair/raw/master/kor_pair_train.csv -P /notebooks/embedding/data
wget https://github.com/songys/Question_pair/raw/master/kor_pair_test.csv -P /notebooks/embedding/data
```



### 데이터 전처리

- `/notebooks/embedding` 위치에서 다음을 실행하면 각 데이터를 전처리할 수 있습니다.
- **네이버 영화 말뭉치 전처리** : 영화 리뷰를 읽어들여 은전한닢(mecab)으로 토크나이즈한 뒤 텍스트 파일로 만듭니다.

```bash
# python preprocess/dump.py nsmc input_file_path output_file_path
python preprocess/dump.py nsmc /notebooks/embedding/data/ratings.txt /notebooks/embedding/data/processed_ratings.txt
```

- **한국어 위키피디아 말뭉치 전처리** : 위키피디아 원문에서 이메일, URL, 여러 형태의 공백 등 불필요 문자를 제거하고 숫자 사이에 공백을 추가하는 등의 전처리를 시행합니다. 이 단계에서는 별도로 토크나이즈를 하진 않습니다.

```bash
# python preprocess/dump.py wiki input_file_path output_file_path
python preprocess/dump.py wiki /notebooks/embedding/data/kowiki-latest-pages-articles.xml.bz2 /notebooks/embedding/data/wiki_ko_raw.txt
```

- **KorSquad 데이터 전처리** : json 내 context를 문서 하나로 취급해 텍스트 파일 형태로 저장합니다. question, anwers은 두 쌍을 공백으로 묶어서 문서 하나로 취급한 후 텍스트 파일로 떨굽니다. 이 단계에서는 별도로 토크나이즈를 하진 않습니다.

```bash
# python preprocess/dump.py korsquad input_file_path output_file_path
python preprocess/dump.py korsquad /notebooks/embedding/data/KorQuAD_v1.0_train.json /notebooks/embedding/data/processed_korsquad_train.txt
```

- **supervised tokenizer** : 은전한닢 등 5개의 한국어 형태소 분석기를 지원합니다. 입력 파일은 한 라인이 한 문서인 형태여야 합니다. 사용법은 다음과 같습니다.

| 형태소 분석기 | 명령                                                         |
| ------------- | ------------------------------------------------------------ |
| 은전한닢      | python preprocess/supervised_nlputils.py mecab input_file_path output_file_path |
| 코모란        | python preprocess/supervised_nlputils.py komoran input_file_path output_file_path |
| Okt           | python preprocess/supervised_nlputils.py okt input_file_path output_file_path |
| 한나눔        | python preprocess/supervised_nlputils.py hannanum input_file_path output_file_path |
| Khaiii        | python preprocess/supervised_nlputils.py khaiii input_file_path output_file_path |

- **unsupervised tokenizer** : soynlp와 구글 SentencePiece 두 가지 분석기를 지원합니다. supervised tokenizer들과 달리 말뭉치의 통계량을 확인한 뒤 토크나이즈를 하기 때문에 토크나이즈 적용 전 모델 학습이 필요합니다.

  (1) soynlp : 사용법은 다음과 같습니다.

  ```bash
  # train : python preprocess/unsupervised_nlputils.py compute_soy_word_score input_file_path model_save_path
  python preprocess/unsupervised_nlputils.py compute_soy_word_score /notebooks/embedding/data/processed_ratings.txt /notebooks/embedding/data/soyword.model
  # tokenize : python preprocess/unsupervised_nlputils.py soy_tokenize input_corpus_path model_save_path output_corpus_path
  python preprocess/unsupervised_nlputils.py soy_tokenize /notebooks/embedding/data/processed_ratings_corpus.txt /notebooks/embedding/data/soyword.model data/tokenized_corpus_soynlp.txt
  ```

  (2) sentencepiece : 사용법은 다음과 같습니다.

  ```bash
  # train
  spm_train --input=/notebooks/embedding/data/processed_ratings_corpus.txt --model_prefix=sentpiece --vocab_size=10000
  # post-process : python preprocess/unsupervised_nlputils.py process_sp_vocab model_vocab_path output_vocab_path
  mv /notebooks/embedding/sentpiece.model /notebooks/embedding/data
  mv /notebooks/embedding/sentpiece.vocab /notebooks/embedding/data
  python preprocess/unsupervised_nlputils.py process_sp_vocab /notebooks/embedding/sentpiece.vocab /notebooks/embedding/processd_sentpiece.vocab
  # tokenize : python preprocess/unsupervised_nlputils.py sentencepiece_tokenize output_vocab_path input_corpus_path output_corpus_path
  python preprocess/unsupervised_nlputils.py sentencepiece_tokenize /notebooks/embedding/data/processd_sentpiece.vocab /notebooks/embedding/data/corrected_ratings_corpus.txt /notebooks/embedding/data/tokenized_corpus_sentpiece.txt
  ```

- **띄어쓰기 교정** : supervised tokenizer이든, unsupervised tokenizer이든 띄어쓰기가 잘 되어 있을 수록 형태소 분석 품질이 좋아집니다. soynlp의 띄어쓰기 교정 모델을 학습하고 말뭉치에 모델을 적용하는 과정은 다음과 같습니다.

  ```bash
  # train : python preprocess/unsupervised_nlputils.py train_space input_corpus_path model_save_path
  python preprocess/unsupervised_nlputils.py train_space /notebooks/embedding/data/processed_ratings.txt /notebooks/embedding/data/space.model
  # correct : python preprocess/unsupervised_nlputils.py input_corpus_path model_save_path output_corpus_path
  python preprocess/unsupervised_nlputils.py apply_space_correct /notebooks/embedding/data/processed_ratings.txt /notebooks/embedding/data/space.model /notebooks/embedding/data/corrected_ratings_corpus.txt
  ```

  