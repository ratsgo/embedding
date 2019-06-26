---
layout: default
---

## 개요
자연언어처리의 근간이 되는 각종 임베딩 기법들에 관련한 튜토리얼입니다. 
한국어 처리를 염두에 두고 작성됐습니다.
아래 모든 명령은 도커 컨테이너 상에서 동작합니다.
개발환경 설정과 관련해선 [이 글](./environment.html)을 참조하세요.


## 데이터 전처리

데이터 전처리는 [이 글](./preprocess.html)을 참조하세요.
형태소 분석은 [이 글](./tokenize.html)을 참조하세요.
다음 한 줄이면 위 글의 내용을 실행하지 않아도 이미 처리된 파일들을 한꺼번에 내려받을 수 있습니다. 

```bash
gdrive_download 1vXiJr0qy_qA-bX4TxmDVqx1VB7_jRcIQ /notebooks/embedding/data/tokenized.zip
```


## 단어 임베딩 모델 학습

`/notebooks/embedding` 위치에서 다음을 실행하면 각 단어 임베딩 모델을 학습할 수 있습니다. 
각 모델의 입력파일은 (1) 한 라인이 하나의 문서 형태이며 (2) 모두 형태소 분석이 완료되어 있어야 합니다. 
명령이 여러 라인으로 구성되어 있을 경우 반드시 순서대로 실행하여야 합니다.


### Latent Semantic Analysis

Word-Context 혹은 PPMI Matrix에 Singular Value Decomposition을 시행합니다.

```bash
mkdir -p /notebooks/embedding/data/word-embeddings/lsa
python models/word_utils.py --method latent_semantic_analysis \
	--input_path /notebooks/embedding/data/tokenized/for-lsa-mecab.txt \
	--output_path /notebooks/embedding/data/word-embeddings/lsa/lsa
```

### Word2Vec

```bash
mkdir -p /notebooks/embedding/data/word-embeddings/word2vec
python models/word_utils.py --method train_word2vec \
	--input_path /notebooks/embedding/data/tokenized/corpus_mecab.txt \
	--output_path /notebooks/embedding/data/word-embeddings/word2vec/word2vec
```

### GloVe

```bash
mkdir -p /notebooks/embedding/data/word-embeddings/glove
/notebooks/embedding/models/glove/build/vocab_count -min-count 5 -verbose 2 < /notebooks/embedding/data/tokenized/corpus_mecab.txt > /notebooks/embedding/data/word-embeddings/glove/glove.vocab
/notebooks/embedding/models/glove/build/cooccur -memory 10.0 -vocab-file /notebooks/embedding/data/word-embeddings/glove/glove.vocab -verbose 2 -window-size 15 < /notebooks/embedding/data/tokenized/corpus_mecab.txt > /notebooks/embedding/data/word-embeddings/glove/glove.cooc
/notebooks/embedding/models/glove/build/shuffle -memory 10.0 -verbose 2 < /notebooks/embedding/data/word-embeddings/glove/glove.cooc > /notebooks/embedding/data/word-embeddings/glove/glove.shuf
/notebooks/embedding/models/glove/build/glove -save-file /notebooks/embedding/data/word-embeddings/glove/glove.vecs -threads 4 -input-file /notebooks/embedding/data/word-embeddings/glove/glove.shuf -x-max 10 -iter 15 -vector-size 100 -binary 2 -vocab-file /notebooks/embedding/data/word-embeddings/glove/glove.vocab -verbose 2
```

### FastText

```bash
mkdir -p /notebooks/embedding/data/word-embeddings/fasttext
/notebooks/embedding/models/fastText/fasttext skipgram -input /notebooks/embedding/data/tokenized/corpus_mecab.txt -output /notebooks/embedding/data/word-embeddings/fasttext/fasttext
```

### Swivel

아래 `swivel.py` 를 실행할 때는 Nvidia-GPU가 있는 환경이면 학습을 빠르게 진행할 수 있습니다.

```bash
mkdir -p /notebooks/embedding/data/word-embeddings/swivel
/notebooks/embedding/models/swivel/fastprep --input /notebooks/embedding/data/tokenized/corpus_mecab.txt --output_dir /notebooks/embedding/data/word-embeddings/swivel/swivel.data
python /notebooks/embedding/models/swivel/swivel.py --input_base_path /notebooks/embedding/data/word-embeddings/swivel/swivel.data --output_base_path /notebooks/embedding/data/word-embeddings/swivel --dim 100
  ```

## 단어 임베딩 다운로드

다음 한 줄이면 위 명령을 수행하지 않고도 이미 학습된 모델들을 한꺼번에 내려받을 수 있습니다. 

```bash
gdrive_download 1yHGtccC2FV3_d6C6_Q4cozYSOgA7bG-e /notebooks/embedding/data/word-embeddings.zip
```


## 단어 임베딩 모델 평가

아래는 단어 임베딩 모델 평가 코드입니다. 파이썬 콘솔에서 실행합니다.

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


## 문장 임베딩 모델 학습

`/notebooks/embedding` 위치에서 다음을 실행하면 각 문장 임베딩 모델을 학습할 수 있습니다. 
각 모델의 입력파일은 (1) 한 라인이 하나의 문서 형태이며 (2) 모두 형태소 분석이 완료되어 있어야 합니다. 
명령이 여러 라인으로 구성되어 있을 경우 반드시 순서대로 실행하여야 합니다.


### Latent Semantic Analysis

TF-IDF Matrix에 Singular Value Decomposition을 시행합니다.

```bash
mkdir -p /notebooks/embedding/data/sentence-embeddings/lsa-tfidf
python models/sent_utils.py --method latent_semantic_analysis \
	--input_path /notebooks/embedding/data/processed/processed_blog.txt \
	--output_path /notebooks/embedding/data/sentence-embeddings/lsa-tfidf/lsa-tfidf.vecs
```

### Doc2Vec

```bash
mkdir -p /notebooks/embedding/data/sentence-embeddings/doc2vec
python models/sent_utils.py --method doc2vec \
	--input_path /notebooks/embedding/data/processed/processed_review_movieid.txt \
	--output_path /notebooks/embedding/data/sentence-embeddings/doc2vec/doc2vec.model
```

### Latent Dirichlet Allocation

```bash
mkdir -p /notebooks/embedding/data/sentence-embeddings/lda
python models/sent_utils.py --method latent_dirichlet_allocation \
	--input_path /notebooks/embedding/data/processed/corrected_ratings_corpus.txt \
	--output_path /notebooks/embedding/data/sentence-embeddings/lda/lda
```

### ELMo

다음을 실행하면 프리트레인(pretrain)을 수행할 수 있습니다.

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

프리트레인(pretrain)이 끝나면 파인튜닝(fine-tuning) 용도로 파라메터를 별도로 저장합니다.

```bash
python models/sent_utils.py --method dump_elmo_weights \
	--input_path /notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt \
	--output_path /notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt/elmo.model
```

아래를 실행해 내 데이터에 맞게 파인튜닝합니다.

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

### BERT

vocabulary를 만듭니다

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

pretrain 학습데이터를 만듭니다. 학습데이터는 tf.record 형태로 저장됩니다.

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

프리트레인(pretrain)을 수행합니다.

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

프리트레인된 모델을 바탕으로 내 데이터에 맞게 튜닝합니다.

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

## 문장 임베딩 다운로드

다음 한 줄이면 위 명령을 수행하지 않고도 이미 학습된 모델들을 한꺼번에 내려받을 수 있습니다. 

```bash
gdrive_download 1u80kFwKlkN8ds0JvQtn6an2dl9QY1rE1 /notebooks/embedding/data/sentence-embeddings.zip
```


## 문장 임베딩 모델 평가

아래는 문장 임베딩 모델 평가 예시 코드입니다.

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