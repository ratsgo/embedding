# embedding
자연언어처리의 근간이 되는 각종 임베딩 기법들에 관련한 튜토리얼입니다.


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