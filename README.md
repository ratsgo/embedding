# embedding tutorials
본 레파지토리는 자연언어처리의 근간이 되는 각종 임베딩 기법들에 관련한 튜토리얼입니다. 한국어 처리를 염두에 두고 작성됐습니다. 본 레파지토리에 있는 코드를 실행하면 `corpus preprocess`, `embedding`, `fine-tuning` 등을 수행할 수 있습니다. 이 모든 과정을 안내하는 튜토리얼 페이지는 다음과 같습니다. 



- http://ratsgo.github.io/embedding



### book

본 튜토리얼은 다음 도서를 보완하기 위해 작성됐습니다. 도서를 구매하지 않아도 튜토리얼 수행에 문제는 없으나 일부 내용은 도서를 참고해야 그 맥락을 완전하게 이해할 수 있습니다. 다음 그림을 클릭하면 도서 구매 사이트로 이동합니다.

<a href="http://www.yes24.com/Product/Goods/78569687"><img src="https://i.imgur.com/j03ENCc.jpg" width="500px" title="embeddings" /></a>

- [정오표](https://ratsgo.github.io/embedding/notice.html)

### embedding methods

본 튜토리얼에서 다루는 임베딩 기법은 다음과 같습니다.



- 단어 수준 임베딩
  - Latent Semantic Analysis
  - Word2Vec
  - GloVe
  - FastText
  - Swivel
- 문장 수준 임베딩
  - Weighted Embeddings
  - Latent Semantic Analysis
  - Latent Dirichlet Allocation
  - Doc2Vec
  - Embeddings from Language Models (ELMo)
  - Bidirectional Encoder Representations from Transformer (BERT)





### corpus preprocess

임베딩 학습데이터를 만들기 위해서는 전처리(preprocess)를 해야 합니다. 본 튜토리얼에서 다루는 오픈소스 패키지는 다음과 같습니다.



- KoNLPy : http://konlpy.org
- Khaiii : https://github.com/kakao/khaiii
- soynlp : https://github.com/lovit/soynlp
- sentencepiece : https://github.com/google/sentencepiece





### embedding fine-tuning

[네이버 영화 리뷰 말뭉치(NSMC)](https://github.com/e9t/nsmc)를 가지고 임베딩을 파인튜닝하는 방법을 실습합니다. 영화 댓글(문서)를 입력으로 하고 긍/부정 극성(polarity)을 분류하는 태스크를 수행합니다. 본 튜토리얼에서 다루는 임베딩 파인튜닝 기법은 다음과 같습니다.



- 문장 수준 임베딩 활용 : Word2Vec, FastText, Swivel + Bi-LSTM with attention layer
- ELMo 활용 : ELMo layer + Bi-LSTM with attention layer
- BERT 활용 : BERT layer + Fully-connected layer



### code

본 레파지토리의 디렉토리 및 코드 구조는 다음과 같습니다.

- docker : 도커 환경 구성을 위한 `Dockerfile`이 있습니다. CPU, GPU 버전을 구분합니다.
- docs : 튜토리얼 페이지와 관련한 마크다운 문서 등이 있습니다.
- models : 임베딩 기법 관련 핵심 코드가 모여 있습니다.
  - bert : BERT 모델 (저자 original 코드)
  - bilm : ELMo 모델 (저자 original 코드)
  - swivel : Swivel 모델 (저자 original 코드)
  - xlnet : XLNet 모델 (저자 original 코드)
  - sent_eval.py : 문장 임베딩 평가 코드
  - sent_utils.py : 문장 임베딩 학습 관련 유틸
  - train_elmo.py : ELMo 프리트레인 코드 (저자 original 코드, 하이퍼파라미터 일부 수정)
  - tune_utils.py : 임베딩 파인튜닝 관련 유틸
  - visualize_utils.py : 임베딩 시각화 관련 유틸
  - word_eval.py : 단어 임베딩 평가 코드
  - word_utils.py : 단어 임베딩 학습 관련 유틸
- preprocess : 말뭉치 전처리 관련 코드가 모여 있습니다.
  - dump.py : 원시 말뭉치(raw corpus)를 1개 라인(line)이 1개 문서인 순수 텍스트 파일로 변환하는 유틸
  - mecab-user-dic.csv : 은전한닢(mecab) 형태소 분석기의 사용자 사전을 추가하기 위한 입력 파일
  - supervised_nlputils.py : KoNLPy, Khaiii 등 지도학습 기반 형태소 분석기 유틸
  - unsupervised_nlputils.py : soynlp, sentencepiece 등 비지도학습 기반 형태소 분석기 유틸
- preprocess.sh : 말뭉치 전처리 자동화 스크립트 모음
- sentmodel.sh : 문장 수준 임베딩 자동화 스크립트 모음
- wordmodel.sh : 단어 수준 임베딩 자동화 스크립트 모음


### environment

본 레파지토리를 수행하기 위한 최적 환경은 도커(docker)입니다. 자세한 내용은 아래를 참고하세요.

- [https://ratsgo.github.io/embedding/environment.html](https://ratsgo.github.io/embedding/environment.html)

구글 코랩(colab) 등 도커 이외에서 수행해야 하는 경우도 있을 수 있습니다.
위의 도커 환경을 구성할 때 썼던 도커파일(dockerfile)을 참고하시면 좋을 것 같습니다. 
CPU, GPU 환경이 각각 다르니 참고에 주의해 주세요!
기본적으로는 tensorflow:1.12.0 버전을 사용하며 아래에 기재된 패키지 버전 이외의 수행은 동작을 보장할 수 없습니다.

- [https://github.com/ratsgo/embedding/tree/master/docker](https://github.com/ratsgo/embedding/tree/master/docker)
