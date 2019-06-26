# embedding
자연언어처리의 근간이 되는 각종 임베딩 기법들에 관련한 튜토리얼입니다. 한국어 처리를 염두에 두고 작성됐습니다.





### 1. 요구사항

docker 혹은 Nvidia-docker2 : [설치방법](https://hiseon.me/2018/02/19/install-docker/)





### 2. 도커 실행방법

튜토리얼을 위한 도커 컨테이너를 실행하려면 자신의 환경에 따라 다음 네 가지 중 하나의 작업을 실시하여야 합니다.

#### 로컬에 nvidia GPU가 있고 Dockerfile로부터 도커이미지를 처음부터 만들어서 컨테이너 띄우기

  ```bash
  git clone https://github.com/ratsgo/embedding.git
  cd embedding
  docker build -t ratsgo/embedding-gpu:1.0 -f docker/Dockerfile-GPU .
  docker run -it --rm --runtime=nvidia ratsgo/embedding-gpu:1.0 bash
  ```

  (2) 로컬에 nvidia GPU가 있고 이미 만들어진 도커이미지를 다운로드 해서 컨테이너 띄우기

  ```bash
  docker pull ratsgo/embedding-gpu:1.0
  docker run -it --rm --runtime=nvidia ratsgo/embedding-gpu:1.0 bash
  ```

  (3) 로컬에 nvidia GPU가 없고 Dockerfile로부터 도커이미지를 처음부터 만들어서 컨테이너 띄우기

  ```bash
  git clone https://github.com/ratsgo/embedding.git
  cd embedding
  docker build -t ratsgo/embedding-cpu:1.0 -f docker/Dockerfile-CPU .
  docker run -it --rm ratsgo/embedding-cpu:1.0 bash
  ```

  (4) 로컬에 nvidia GPU가 없고 이미 만들어진 도커이미지를 다운로드 해서 컨테이너 띄우기

  ```bash
  docker pull ratsgo/embedding-cpu:1.0
  docker run -it --rm ratsgo/embedding-cpu:1.0 bash
  ```

- 아래 모든 명령은 도커 컨테이너 상에서 동작합니다.





### 3. 데이터 덤프

- 아래의 `wget` 명령어를 입력해 필요한 말뭉치를 다운로드합니다.

- [네이버 영화 말뭉치](https://github.com/e9t/nsmc)

  ```bash
  wget https://github.com/e9t/nsmc/raw/master/ratings.txt -P /notebooks/embedding/data/raw
  wget https://github.com/e9t/nsmc/raw/master/ratings_train.txt -P /notebooks/embedding/data/raw
  wget https://github.com/e9t/nsmc/raw/master/ratings_test.txt -P /notebooks/embedding/data/raw
  ```

- 한국어 위키피디아

  ```bash
  wget https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2 -P /notebooks/embedding/data/raw
  ```

- [KorQuAD](https://korquad.github.io)

  ```bash
  wget https://korquad.github.io/dataset/KorQuAD_v1.0_train.json -P /notebooks/embedding/data/raw
  wget https://korquad.github.io/dataset/KorQuAD_v1.0_dev.json -P /notebooks/embedding/data/raw
  ```

- [유사 문장](https://github.com/songys/Question_pair)

  ```bash
  wget https://github.com/songys/Question_pair/raw/master/kor_pair_train.csv -P /notebooks/embedding/data/raw
  wget https://github.com/songys/Question_pair/raw/master/kor_Pair_test.csv -P /notebooks/embedding/data/raw
  ```

- [ratsgo blog](http://ratsgo.github.io)

  ```bash
  function gdrive_download () {
    CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
    rm -rf /tmp/cookies.txt
  }
          
  gdrive_download 1Few7-Mh3JypQN3rjnuXD8yAXrkxUwmjS /notebooks/embedding/data/processed/processed_blog.txt
  ```

  



### 4. 데이터를 문장 단위 텍스트 파일로 저장하기

- `/notebooks/embedding` 위치에서 다음을 실행하면 각기 다른 형식의 데이터를 한 라인이 한 문서인 형태의 텍스트 파일로 저장합니다. 이 단계에서는 별도로 토크나이즈를 하진 않습니다.

- **네이버 영화 말뭉치 전처리**

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

- **한국어 위키피디아 말뭉치 전처리**

  위키피디아 원문에서 이메일, URL, 여러 형태의 공백 등 불필요 문자를 제거하고 숫자 사이에 공백을 추가하는 등의 전처리를 시행합니다. 

  ```bash
  python preprocess/dump.py --preprocess_mode wiki \
  	--input_path /notebooks/embedding/data/raw/kowiki-latest-pages-articles.xml.bz2 \
  	--output_path /notebooks/embedding/data/processed/processed_wiki_ko.txt
  ```

- **KorSquad 데이터 전처리**

  json 내 context를 문서 하나로 취급합니다. question, anwers은 두 쌍을 공백으로 묶어서 문서 하나로 취급합니다.

  ```bash
  python preprocess/dump.py --preprocess_mode korsquad \
  	--input_path /notebooks/embedding/data/raw/KorQuAD_v1.0_train.json \
  	--output_path /notebooks/embedding/data/processed/processed_korsquad_train.txt
  python preprocess/dump.py --preprocess_mode korsquad \
  	--input_path /notebooks/embedding/data/raw/KorQuAD_v1.0_dev.json \
  	--output_path data/processed/processed_korsquad_dev.txt
  ```

- 다음 한 줄이면 위 명령을 수행하지 않고도 이미 처리된 파일들을 한꺼번에 내려받을 수 있습니다. 

  ```bash
  gdrive_download 1hscU5_f_1vXfbhHabNpqfnp8DU2ZWmcT /notebooks/embedding/data/processed.zip
  ```

  



### 5. 형태소 분석

- `/notebooks/embedding` 위치에서 다음을 실행하면 각 데이터를 형태소 분석할 수 있습니다. 입력 파일은 한 라인이 한 문서인 형태여야 합니다. 

- **supervised tokenizer**

  은전한닢 등 5개의 한국어 형태소 분석기를 지원합니다. 사용법은 다음과 같습니다.

  | 형태소 분석기 | 명령                                                         |
  | ------------- | ------------------------------------------------------------ |
  | 은전한닢      | python preprocess/supervised_nlputils.py --tokenizer mecab --input_path input_file_path --output_path output_file_path |
  | 코모란        | python preprocess/supervised_nlputils.py --tokenizer komoran --input_path input_file_path --output_path output_file_path |
  | Okt           | python preprocess/supervised_nlputils.py --tokenizer okt --input_path input_file_path --output_path output_file_path |
  | 한나눔        | python preprocess/supervised_nlputils.py --tokenizer hannanum --input_path input_file_path --output_path output_file_path |
  | Khaiii        | python preprocess/supervised_nlputils.py --tokenizer khaiii --input_path input_file_path --output_path output_file_path |

- **unsupervised tokenizer**

  soynlp와 구글 SentencePiece 두 가지 분석기를 지원합니다. supervised tokenizer들과 달리 말뭉치의 통계량을 확인한 뒤 토크나이즈를 하기 때문에 토크나이즈 적용 전 모델 학습이 필요합니다.

  (1) soynlp : 사용법은 다음과 같습니다.

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

  (2) sentencepiece : 사용법은 다음과 같습니다.

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

- **띄어쓰기 교정** : `supervised tokenizer`이든, `unsupervised tokenizer`이든 띄어쓰기가 잘 되어 있을 수록 형태소 분석 품질이 좋아집니다. soynlp의 띄어쓰기 교정 모델을 학습하고 말뭉치에 모델을 적용하는 과정은 다음과 같습니다.

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


- 다음 한 줄이면 위 명령을 수행하지 않고도 이미 처리된 파일들을 한꺼번에 내려받을 수 있습니다. 

  ```bash
  gdrive_download 1vXiJr0qy_qA-bX4TxmDVqx1VB7_jRcIQ /notebooks/embedding/data/tokenized.zip
  ```

  



### 6. 단어 임베딩 모델 학습

- `/notebooks/embedding` 위치에서 다음을 실행하면 각 단어 임베딩 모델을 학습할 수 있습니다. 각 모델의 입력파일은 (1) 한 라인이 하나의 문서 형태이며 (2) 모두 형태소 분석이 완료되어 있어야 합니다. 명령이 여러 라인으로 구성되어 있을 경우 반드시 순서대로 실행하여야 합니다.

- **데이터 준비** : 5장에서 설명드린 것처럼, 이미 형태소 분석이 완료된 파일들을 내려 받아 준비합니다. 이후 아래처럼 데이터를 merge합니다.

  ```bash
  gdrive_download 1vXiJr0qy_qA-bX4TxmDVqx1VB7_jRcIQ /notebooks/embedding/data/tokenized.zip
  cat /notebooks/embedding/data/tokenized/wiki_ko_mecab.txt /notebooks/embedding/data/tokenized/ratings_mecab.txt /notebooks/embedding/data/tokenized/korsquad_mecab.txt > /notebooks/embedding/data/tokenized/corpus_mecab.txt
  cat /notebooks/embedding/data/tokenized/ratings_mecab.txt /notebooks/embedding/data/tokenized/korsquad_mecab.txt > /notebooks/embedding/data/tokenized/for-lsa-mecab.txt
  ```

- **Latent Semantic Analysis** : Word-Context Matrix에 Singular Value Decomposition을 시행합니다.

  ```bash
  mkdir -p /notebooks/embedding/data/word-embeddings/lsa
  python models/word_utils.py --method latent_semantic_analysis \
  	--input_path /notebooks/embedding/data/tokenized/for-lsa-mecab.txt \
  	--output_path /notebooks/embedding/data/word-embeddings/lsa/lsa
  ```

- **Word2Vec**

  ```bash
  mkdir -p /notebooks/embedding/data/word-embeddings/word2vec
  python models/word_utils.py --method train_word2vec \
  	--input_path /notebooks/embedding/data/tokenized/corpus_mecab.txt \
  	--output_path /notebooks/embedding/data/word-embeddings/word2vec/word2vec
  ```

- **GloVe**

  ```bash
  mkdir -p /notebooks/embedding/data/word-embeddings/glove
  /notebooks/embedding/models/glove/build/vocab_count -min-count 5 -verbose 2 < /notebooks/embedding/data/tokenized/corpus_mecab.txt > /notebooks/embedding/data/word-embeddings/glove/glove.vocab
  /notebooks/embedding/models/glove/build/cooccur -memory 10.0 -vocab-file /notebooks/embedding/data/word-embeddings/glove/glove.vocab -verbose 2 -window-size 15 < /notebooks/embedding/data/tokenized/corpus_mecab.txt > /notebooks/embedding/data/word-embeddings/glove/glove.cooc
  /notebooks/embedding/models/glove/build/shuffle -memory 10.0 -verbose 2 < /notebooks/embedding/data/word-embeddings/glove/glove.cooc > /notebooks/embedding/data/word-embeddings/glove/glove.shuf
  /notebooks/embedding/models/glove/build/glove -save-file /notebooks/embedding/data/word-embeddings/glove/glove.vecs -threads 4 -input-file /notebooks/embedding/data/word-embeddings/glove/glove.shuf -x-max 10 -iter 15 -vector-size 100 -binary 2 -vocab-file /notebooks/embedding/data/word-embeddings/glove/glove.vocab -verbose 2
  ```

- **FastText**

  ```bash
  mkdir -p /notebooks/embedding/data/word-embeddings/fasttext
  /notebooks/embedding/models/fastText/fasttext skipgram -input /notebooks/embedding/data/tokenized/corpus_mecab.txt -output /notebooks/embedding/data/word-embeddings/fasttext/fasttext
  ```

- **Swivel** : `swivel.py` 를 실행할 때는 Nvidia-GPU가 있는 환경이면 학습을 빠르게 진행할 수 있습니다.

  ```bash
  mkdir -p /notebooks/embedding/data/word-embeddings/swivel
  /notebooks/embedding/models/swivel/fastprep --input /notebooks/embedding/data/tokenized/corpus_mecab.txt --output_dir /notebooks/embedding/data/word-embeddings/swivel/swivel.data
  python /notebooks/embedding/models/swivel/swivel.py --input_base_path /notebooks/embedding/data/word-embeddings/swivel/swivel.data --output_base_path /notebooks/embedding/data/word-embeddings/swivel --dim 100
  ```

- 다음 한 줄이면 위 명령을 수행하지 않고도 이미 학습된 모델들을 한꺼번에 내려받을 수 있습니다. 

  ```bash
  gdrive_download 1yHGtccC2FV3_d6C6_Q4cozYSOgA7bG-e /notebooks/embedding/data/word-embeddings.zip
  ```






### 7. 단어 임베딩 모델 평가

- 아래는 단어 임베딩 모델 평가 예시 코드입니다.

  ```python
  from models.word_eval import WordEmbeddingEval
  model = WordEmbeddingEval(vecs_fname="word2vec_path", method="word2vec")
  model.word_sim_test("data/kor_ws353.csv")
  model.word_analogy_test("data/kor_analogy_semantic.txt")
  model.word_analogy_test("data/kor_analogy_syntactic.txt")
  model.most_similar("문재인")
  model.visualize_words("data/kor_analogy_semantic.txt", palette="Viridis256")
  model.visualize_between_words("data/kor_analogy_semantic.txt", palette="Greys256")
  ```

- 아래는 위의 코드 실행으로 시각화된 결과물의 예시입니다.

<img src='http://drive.google.com/uc?export=view&id=1Aia5psgzxU2kMiW7-3jOG5CM4bp0jox9' width=700 />

<img src='http://drive.google.com/uc?export=view&id=1aCLrdqDZvChepoDFZ4oBmqsgvJ6GJwNL' widh=700 />



### 8. 문장 임베딩 모델 학습

- `/notebooks/embedding` 위치에서 다음을 실행하면 각 문장 임베딩 모델을 학습할 수 있습니다. 각 모델의 입력파일은 (1) 한 라인이 하나의 문서 형태이며 (2) 모두 형태소 분석이 완료되어 있어야 합니다. 명령이 여러 라인으로 구성되어 있을 경우 반드시 순서대로 실행하여야 합니다.

- **데이터 준비** : 4장과 5장에서 이미 전처리가 완료된 말뭉치를 내려받습니다.

  ```
  gdrive_download 1hscU5_f_1vXfbhHabNpqfnp8DU2ZWmcT /notebooks/embedding/data/processed.zip
  gdrive_download 1vXiJr0qy_qA-bX4TxmDVqx1VB7_jRcIQ /notebooks/embedding/data/tokenized.zip
  ```

- **Latent Semantic Analysis** : TF-IDF Matrix에 Singular Value Decomposition을 시행합니다.

  ```bash
  mkdir -p /notebooks/embedding/data/sentence-embeddings/lsa-tfidf
  python models/sent_utils.py --method latent_semantic_analysis \
  	--input_path /notebooks/embedding/data/processed/processed_blog.txt \
  	--output_path /notebooks/embedding/data/sentence-embeddings/lsa-tfidf/lsa-tfidf.vecs
  ```

- **Doc2Vec**

  ```bash
  mkdir -p /notebooks/embedding/data/sentence-embeddings/doc2vec
  python models/sent_utils.py --method doc2vec \
  	--input_path /notebooks/embedding/data/processed/processed_review_movieid.txt \
  	--output_path /notebooks/embedding/data/sentence-embeddings/doc2vec/doc2vec.model
  ```

- **Latent Dirichlet Allocation**

  ```bash
  mkdir -p /notebooks/embedding/data/sentence-embeddings/lda
  python models/sent_utils.py --method latent_dirichlet_allocation \
  	--input_path /notebooks/embedding/data/processed/corrected_ratings_corpus.txt \
  	--output_path /notebooks/embedding/data/sentence-embeddings/lda/lda
  ```

- **ELMo**

  (1) pretrain

  ```bash
  # preprocess
  mkdir -p /notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt/traindata
  cat /notebooks/embedding/data/tokenized/wiki_ko_mecab.txt /notebooks/embedding/data/tokenized/ratings_mecab.txt /notebooks/embedding/data/tokenized/korsquad_mecab.txt > /notebooks/embedding/data/tokenized/corpus_mecab.txt
  split -l 100000 /notebooks/embedding/data/tokenized/corpus_mecab.txt /notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt/traindata/data_
  # make vocab
  export LC_CTYPE=C.UTF-8
  python models/sent_utils.py --method construct_elmo_vocab \
  	--input_path /notebooks/embedding/data/tokenized/corpus_mecab.txt \
  	--output_path /notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt/elmo-vocab.txt
  # pretrain
  nohup sh -c "python models/train_elmo.py \
  	--train_prefix='/notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt/traindata/*' \
  	--vocab_file /notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt/elmo-vocab.txt \
  	--save_dir /notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt \
  	--n_gpus 1" > elmo-pretrain.log &
  ```

  (2) pretrain된 모델의 파라메터를 덤프합니다.

  ```bash
  python models/sent_utils.py --method dump_elmo_weights \
  	--input_path /notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt \
  	--output_path /notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt/elmo.model
  ```

  (3) fine-tune : (2)에서 덤프한 파라메터를 바탕으로 내 데이터에 맞게 튜닝합니다.

  ```bash
  export LC_CTYPE=C.UTF-8
  nohup sh -c "python models/tune_utils.py --model_name elmo \
  	--train_corpus_fname /notebooks/embedding/data/processed/processed_ratings_train.txt \
  	--test_corpus_fname /notebooks/embedding/data/processed/processed_ratings_test.txt \
  	--vocab_fname /notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt/elmo-vocab.txt \
  	--pretrain_model_fname /notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt/elmo.model \
  	--config_fname /notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt/options.json \
  	--model_save_path /notebooks/embedding/data/sentence-embeddings/elmo/tune-ckpt" > elmo-tune.log &
  ```

- **BERT**

  (1) vocabulary를 만듭니다

  ```bash
  mkdir -p /notebooks/embedding/data/sentence-embeddings/bert/pretrain-ckpt/vocab
  cd  /notebooks/embedding/data/sentence-embeddings/bert/pretrain-ckpt/vocab
  spm_train --input=/notebooks/embedding/data/processed/corrected_ratings_corpus.txt --model_prefix=sentpiece --vocab_size=32000
  cd  /notebooks/embedding
  export LC_CTYPE=C.UTF-8
  python preprocess/unsupervised_nlputils.py --preprocess_mode process_sp_vocab \
  	--input_path /notebooks/embedding/data/sentence-embeddings/bert/pretrain-ckpt/vocab/sentpiece.vocab \
  	--vocab_path /notebooks/embedding/data/sentence-embeddings/bert/pretrain-ckpt/vocab.txt
  ```

  (2) pretrain 학습데이터를 만듭니다 : 학습데이터는 tf.record 형태로 저장됩니다.

  ```bash
  export LC_CTYPE=C.UTF-8
  mkdir -p /notebooks/embedding/data/sentence-embeddings/bert/pretrain-ckpt/traindata
  python models/bert/create_pretraining_data.py \
  	--input_file=/notebooks/embedding/data/processed/corrected_ratings_corpus.txt \
  	--output_file=/notebooks/embedding/data/sentence-embeddings/bert/pretrain-ckpt/traindata/tfrecord \
  	--vocab_file=/notebooks/embedding/data/sentence-embeddings/bert/pretrain-ckpt/vocab.txt \
  	--do_lower_case=False \
  	--max_seq_length=128 \
  	--max_predictions_per_seq=20 \
  	--masked_lm_prob=0.15 \
  	--random_seed=7 \
  	--dupe_factor=5
  ```

  (3) pretrain

  ```bash
  # download bert configurations
  gdrive_download 1MAfK-GLVUdlje-Twp6hFEOXYk3LjeIak /notebooks/embedding/data/sentence-embeddings/bert/pretrain-ckpt/bert_config.json
  # pretrain
  export LC_CTYPE=C.UTF-8
  nohup sh -c "python models/bert/run_pretraining.py \
  	--input_file=/notebooks/embedding/data/sentence-embeddings/bert/pretrain-ckpt/traindata/tfrecord* \
  	--output_dir=/notebooks/embedding/data/sentence-embeddings/bert/pretrain-ckpt \
  	--do_train=True \
  	--do_eval=True \
  	--bert_config_file=/notebooks/embedding/data/sentence-embeddings/bert/pretrain-ckpt/bert_config.json \
  	--train_batch_size=32 \
  	--max_seq_length=128 \
  	--max_predictions_per_seq=20 \
  	--learning_rate=2e-5" > bert-pretrain.log &
  ```

  (4) fine-tune : (2)에서 pretrain된 모델을 바탕으로 내 데이터에 맞게 튜닝합니다.

  ```bash
  export LC_CTYPE=C.UTF-8
  nohup sh -c "python models/tune_utils.py --model_name bert \
  	--train_corpus_fname /notebooks/embedding/data/processed/processed_ratings_train.txt \
  	--test_corpus_fname /notebooks/embedding/data/processed/processed_ratings_test.txt \
  	--vocab_fname /notebooks/embedding/data/sentence-embeddings/bert/multi_cased_L-12_H-768_A-12/vocab.txt \
  	--pretrain_model_fname /notebooks/embedding/data/sentence-embeddings/bert/multi_cased_L-12_H-768_A-12/bert_model.ckpt \
  	--config_fname /notebooks/embedding/data/sentence-embeddings/bert/multi_cased_L-12_H-768_A-12/bert_config.json \
  	--model_save_path /notebooks/embedding/data/sentence-embeddings/bert/tune-ckpt" > bert-tune.log &
  ```


- 다음 한 줄이면 위 명령을 수행하지 않고도 이미 학습된 모델들을 한꺼번에 내려받을 수 있습니다. 

  ```bash
  gdrive_download 1u80kFwKlkN8ds0JvQtn6an2dl9QY1rE1 /notebooks/embedding/data/sentence-embeddings.zip
  ```






### 9. 문장 임베딩 모델 평가

- 아래는 문장 임베딩 모델 평가 예시 코드입니다.

```python
from models.sent_eval import BERTEmbeddingEvaluator
model = BERTEmbeddingEvaluator()
model.get_sentence_vector("나는 학교에 간다")
model.get_token_vector_sequence("나는 학교에 간다")
model.visualize_homonym("배", ["배 고프다", "배 나온다", "배가 불렀다",
                                "배는 수분이 많은 과일이다", 
                                "배를 바다에 띄웠다", "배 멀미가 난다"])
model.visualize_self_attention_scores("배 고파 밥줘")
model.predict("이 영화 정말 재미 있다")
model.visualize_between_sentences(sampled_sentences)
model.visualize_sentences(sampled_sentences)
```

- 아래는 위의 코드 실행으로 시각화한 결과물의 예시입니다.

<img src='http://drive.google.com/uc?export=view&id=1J8bsPWMBPVUaRehTlCwZ5-GNQLv6TiqW' width=500>

<img src='http://drive.google.com/uc?export=view&id=1eh18VG1kRU7wdWT7zuG_69HA1r7iS_lV' width=500>

