본 사이트는 자연언어처리의 근간이 되는 각종 임베딩 기법들에 관련한 튜토리얼입니다.  한국어 처리를 염두에 두고 작성됐습니다. 



## 1. 도서 안내

이 튜토리얼은 다음 도서를 보완하기 위해 작성됐습니다. 도서를 구매하지 않아도 튜토리얼 수행에 문제는 없으나 일부 내용은 도서를 참고해야 그 맥락을 완전하게 이해할 수 있습니다. 다음 그림을 클릭하면 도서 구매 사이트로 이동합니다.

<a href="http://www.yes24.com/Product/Goods/78569687"><img src="https://i.imgur.com/j03ENCc.jpg" width="500px" title="embeddings" /></a>



## 2. 개발환경 및 데이터 준비

아래 모든 명령은 도커 컨테이너 상에서 동작합니다. 개발환경 설정과 관련해선 [이 글](./environment.html)을 참조하세요.  
데이터 전처리 방법을 보시려면 [이 글](./preprocess.html)을 참조하세요.   
형태소 분석하는 방법은 [이 글](./tokenize.html)을 참조하세요.  
데이터 다운로드는 [이 글](https://ratsgo.github.io/embedding/downloaddata.html)을 참조하세요.



## 3. 단어 임베딩 모델 학습

`/notebooks/embedding` 위치에서 다음을 실행하면 각 단어 임베딩 모델을 학습할 수 있습니다.  각 모델의 입력파일은 (1) 한 라인이 하나의 문서 형태이며 (2) 모두 형태소 분석이 완료되어 있어야 합니다. 명령이 여러 라인으로 구성되어 있을 경우 반드시 순서대로 실행하여야 합니다.



### 3-1. Latent Semantic Analysis

Word-Context 혹은 PPMI Matrix에 Singular Value Decomposition을 시행합니다. 자신이 가진 데이터(단 형태소 분석이 완료되어 있어야 함)로 수행하고 싶다면 `input_path`를 바꿔주면 됩니다.

```bash
mkdir -p /notebooks/embedding/data/word-embeddings/lsa
python models/word_utils.py --method latent_semantic_analysis \
	--input_path /notebooks/embedding/data/tokenized/for-lsa-mecab.txt \
	--output_path /notebooks/embedding/data/word-embeddings/lsa/lsa
```



### 3-2. Word2Vec

Word2Vec 임베딩을 학습합니다. 자신이 가진 데이터(단 형태소 분석이 완료되어 있어야 함)로 임베딩하고 싶다면 `input_path`를 바꿔주면 됩니다.

```bash
mkdir -p /notebooks/embedding/data/word-embeddings/word2vec
python models/word_utils.py --method train_word2vec \
	--input_path /notebooks/embedding/data/tokenized/corpus_mecab.txt \
	--output_path /notebooks/embedding/data/word-embeddings/word2vec/word2vec
```



### 3-3. GloVe

GloVe 임베딩을 학습합니다. 자신이 가진 데이터(단 형태소 분석이 완료되어 있어야 함)로 임베딩하고 싶다면 아래 스크립트에서 `/notebooks/embedding/data/tokenized/corpus_mecab.txt`를 해당 데이터 경로로 바꿔주면 됩니다.

```bash
mkdir -p /notebooks/embedding/data/word-embeddings/glove
/notebooks/embedding/models/glove/build/vocab_count -min-count 5 -verbose 2 < /notebooks/embedding/data/tokenized/corpus_mecab.txt > /notebooks/embedding/data/word-embeddings/glove/glove.vocab
/notebooks/embedding/models/glove/build/cooccur -memory 10.0 -vocab-file /notebooks/embedding/data/word-embeddings/glove/glove.vocab -verbose 2 -window-size 15 < /notebooks/embedding/data/tokenized/corpus_mecab.txt > /notebooks/embedding/data/word-embeddings/glove/glove.cooc
/notebooks/embedding/models/glove/build/shuffle -memory 10.0 -verbose 2 < /notebooks/embedding/data/word-embeddings/glove/glove.cooc > /notebooks/embedding/data/word-embeddings/glove/glove.shuf
/notebooks/embedding/models/glove/build/glove -save-file /notebooks/embedding/data/word-embeddings/glove/glove.vecs -threads 4 -input-file /notebooks/embedding/data/word-embeddings/glove/glove.shuf -x-max 10 -iter 15 -vector-size 100 -binary 2 -vocab-file /notebooks/embedding/data/word-embeddings/glove/glove.vocab -verbose 2
```



### 3-4. FastText

FastText 임베딩을 학습합니다. 자신이 가진 데이터(단 형태소 분석이 완료되어 있어야 함)로 임베딩하고 싶다면 `input`을 바꿔주면 됩니다.

```bash
mkdir -p /notebooks/embedding/data/word-embeddings/fasttext
/notebooks/embedding/models/fastText/fasttext skipgram -input /notebooks/embedding/data/tokenized/corpus_mecab.txt -output /notebooks/embedding/data/word-embeddings/fasttext/fasttext
```



### 3-5. Swivel

Swivel 임베딩을 학습합니다. 자신이 가진 데이터(단 형태소 분석이 완료되어 있어야 함)로 임베딩하고 싶다면 `input`만 바꿔주면 됩니다. 아래 `swivel.py` 를 실행할 때는 Nvidia-GPU가 있는 환경이면 학습을 빠르게 진행할 수 있습니다.

```bash
mkdir -p /notebooks/embedding/data/word-embeddings/swivel
/notebooks/embedding/models/swivel/fastprep --input /notebooks/embedding/data/tokenized/corpus_mecab.txt --output_dir /notebooks/embedding/data/word-embeddings/swivel/swivel.data
python /notebooks/embedding/models/swivel/swivel.py --input_base_path /notebooks/embedding/data/word-embeddings/swivel/swivel.data --output_base_path /notebooks/embedding/data/word-embeddings/swivel --dim 100
```



## 4. 단어 임베딩 모델 평가

아래는 단어 임베딩 모델 평가 코드입니다. 단, 해당 단어 임베딩이 로컬 디렉토리에 존재해야 합니다. 이미 학습이 완료된 단어 임베딩을 내려받으려면 [이 글](https://ratsgo.github.io/embedding/downloaddata.html)을 참고하세요. 아래 코드는 파이썬 콘솔에서 실행합니다.

```python
from models.word_eval import WordEmbeddingEval
model = WordEmbeddingEval(vecs_fname="word2vec_path", method="word2vec")
model.word_sim_test("data/kor_ws353.csv")
model.word_analogy_test("data/kor_analogy_semantic.txt")
model.word_analogy_test("data/kor_analogy_syntactic.txt")
model.most_similar("희망")
model.visualize_words("data/kor_analogy_semantic.txt", palette="Viridis256")
model.visualize_between_words("data/kor_analogy_semantic.txt", palette="Greys256")
```



## 5. 문장 임베딩 모델 학습

`/notebooks/embedding` 위치에서 다음을 실행하면 각 문장 임베딩 모델을 학습할 수 있습니다. 
각 모델의 입력파일은 (1) 한 라인이 하나의 문서 형태이며 (2) 모두 형태소 분석이 완료되어 있어야 합니다. 
명령이 여러 라인으로 구성되어 있을 경우 반드시 순서대로 실행하여야 합니다.



### 5-1. Latent Semantic Analysis

TF-IDF Matrix에 Singular Value Decomposition을 시행합니다. 자신이 가진 데이터(단 형태소 분석이 완료되어 있어야 함)로 수행하고 싶다면 `input_path`를 바꿔주면 됩니다.

```bash
mkdir -p /notebooks/embedding/data/sentence-embeddings/lsa-tfidf
python models/sent_utils.py --method latent_semantic_analysis \
	--input_path /notebooks/embedding/data/processed/processed_blog.txt \
	--output_path /notebooks/embedding/data/sentence-embeddings/lsa-tfidf/lsa-tfidf.vecs
```



### 5-2. Doc2Vec

Doc2Vec 임베딩을 학습합니다. 자신이 가진 데이터(단 형태소 분석이 완료되어 있어야 함)로 수행하고 싶다면 `input_path`를 바꿔주면 됩니다.

```bash
mkdir -p /notebooks/embedding/data/sentence-embeddings/doc2vec
python models/sent_utils.py --method doc2vec \
	--input_path /notebooks/embedding/data/processed/processed_review_movieid.txt \
	--output_path /notebooks/embedding/data/sentence-embeddings/doc2vec/doc2vec.model
```



### 5-3. Latent Dirichlet Allocation

LDA 임베딩을 학습합니다. 자신이 가진 데이터(단 형태소 분석이 완료되어 있어야 함)로 수행하고 싶다면 `input_path`를 바꿔주면 됩니다.

```bash
mkdir -p /notebooks/embedding/data/sentence-embeddings/lda
python models/sent_utils.py --method latent_dirichlet_allocation \
	--input_path /notebooks/embedding/data/processed/corrected_ratings_corpus.txt \
	--output_path /notebooks/embedding/data/sentence-embeddings/lda/lda
```



### 5-4. ELMo

다음을 실행하면 프리트레인(pretrain)을 수행할 수 있습니다. 자신이 가지고 있는 데이터로 프리트레인을 수행하고 싶다면 `sentmodel.sh`의 `pretrain-elmo` 항목에서 `sent_utils.py`를 실행하는 부분의 `input_path`를 자신이 가진 말뭉치 경로로 바꿔주면 됩니다. 단 해당 말뭉치는 형태소 분석을 모두 마친 데이터여야 합니다.

```bash
git pull origin master
bash sentmodel.sh pretrain-elmo
```

프리트레인(pretrain)이 끝나면 파인튜닝(fine-tuning) 용도로 파라메터를 별도로 저장합니다.

```bash
bash sentmodel.sh dump-pretrained-elmo
```

컴퓨팅 환경이 여의치 않거나 ELMo 프리트레인에 리소스를 투자하고 싶지 않다면 아래 명령을 수행하면 프리트레인이 완료된 ELMo 모델을 내려받을 수 있습니다. 이 모델은 한국어 위키백과, 네이버 영화 리뷰 말뭉치, KorQuAD로 학습됐습니다.

```bash
bash sentmodel.sh download-pretrained-elmo
```

아래를 실행해 내 데이터에 맞게 파인튜닝합니다. 파인튜닝을 수행하려면 프리트레인이 완료된 ELMo 모델이 `/notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt` 경로에 있어야 합니다. 

자신이 가지고 있는 데이터로 파인튜닝을 수행하고 싶다면 `sentmodel.sh`의 `tune-elmo` 항목에서 `train_corpus_fname`과 `test_corpus_fname`을 자신이 가진 말뭉치 경로로 바꿔주면 됩니다. 해당 말뭉치는 형태소 분석이 안된 원시 데이터여야 합니다.

```bash
bash sentmodel.sh tune-elmo
```



### 5-5. BERT

다음을 실행하면 프리트레인(pretrain)을 수행할 수 있습니다. 자신이 가지고 있는 데이터로 프리트레인을 수행하고 싶다면 `sentmodel.sh`의 `pretrain-bert` 항목에서 `dump.py`를 실행하는 부분의 `input_path`를 자신이 가진 말뭉치 경로로 바꿔주면 됩니다. 단 해당 말뭉치는 형태소 분석이 안된 원시 데이터여야 합니다.

```bash
bash sentmodel.sh pretrain-bert
```

컴퓨팅 환경이 여의치 않거나 BERT 프리트레인에 리소스를 투자하고 싶지 않다면 아래 명령을 수행하면 프리트레인이 완료된 BERT 모델을 내려받을 수 있습니다. 이 모델은 자연어 처리 연구자 오연택 님께서 공개한 모델입니다. 자세한 구축 방법은 [이곳](https://github.com/yeontaek/BERT-Korean-Model)을 참고하세요.

```bash
bash sentmodel.sh download-pretrained-bert
```

아래를 실행해 내 데이터에 맞게 파인튜닝합니다. 파인튜닝을 수행하려면 프리트레인이 완료된 BERT 모델이 `/notebooks/embedding/data/sentence-embeddings/bert/pretrain-ckpt` 경로에 있어야 합니다. 

자신이 가지고 있는 데이터로 파인튜닝을 수행하고 싶다면 `sentmodel.sh`의 `tune-bert` 항목에서 `train_corpus_fname`과 `test_corpus_fname`을 자신이 가진 말뭉치 경로로 바꿔주면 됩니다. 해당 말뭉치는 형태소 분석이 안된 원시 데이터여야 합니다.

```bash
bash sentmodel.sh tune-bert
```





## 6. 문장 임베딩 모델 평가

아래는 파인튜닝을 마친 문장 임베딩 모델을 평가하는 코드 예시입니다. 파이썬 콘솔에서 실행하면 됩니다. 단, 모든 파일이 해당 디렉토리에 존재해야 합니다. 

```python
from models.sent_eval import ELMoEmbeddingEvaluator
model = ELMoEmbeddingEvaluator(tune_model_fname="/notebooks/embedding/data/sentence-embeddings/elmo/tune-ckpt",
                               pretrain_model_fname="/notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt/elmo.model",
                               options_fname="/notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt/options.json",
                               vocab_fname="/notebooks/embedding/data/sentence-embeddings/elmo/pretrain-ckpt/elmo-vocab.txt",
                               max_characters_per_token=30, dimension=256, num_labels=2)
model.predict("이 영화 엄청 재미있네요") # label 예측
model.get_token_vector_sequence("이 영화 엄청 재미있네요") # ELMo의 토큰별 임베딩 추출
model.get_sentence_vector("이 영화 엄청 재미있네요") # ELMo의 문장 임베딩 추출
```