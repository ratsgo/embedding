---
layout: default
title: 개발환경 설정
description: 도커 설치 및 컨테이너 실행 방법을 안내합니다.
---

## 개발환경 설정

이 페이지에서는 도커 설치 및 컨테이너 실행 방법을 안내합니다.



### 요구사항

docker 혹은 Nvidia-docker2 : [설치방법](https://hiseon.me/2018/02/19/install-docker/)



### 도커 실행방법

튜토리얼을 위한 도커 컨테이너를 실행하려면 자신의 환경에 따라 다음 네 가지 중 하나의 작업을 실시하여야 합니다.



#### 로컬에 nvidia GPU가 있고 Dockerfile로부터 도커이미지를 처음부터 만들어서 컨테이너 띄우기

```bash
git clone https://github.com/ratsgo/embedding.git
cd embedding
docker build -t ratsgo/embedding-gpu:1.2 -f docker/Dockerfile-GPU .
docker run -it --rm --runtime=nvidia ratsgo/embedding-gpu:1.2 bash
```



#### 로컬에 nvidia GPU가 있고 이미 만들어진 도커이미지를 다운로드 해서 컨테이너 띄우기

```bash
docker pull ratsgo/embedding-gpu:1.2
docker run -it --rm --runtime=nvidia ratsgo/embedding-gpu:1.2 bash
```



#### 로컬에 nvidia GPU가 없고 Dockerfile로부터 도커이미지를 처음부터 만들어서 컨테이너 띄우기

```bash
git clone https://github.com/ratsgo/embedding.git
cd embedding
docker build -t ratsgo/embedding-cpu:1.2 -f docker/Dockerfile-CPU .
docker run -it --rm ratsgo/embedding-cpu:1.2 bash
```



#### 로컬에 nvidia GPU가 없고 이미 만들어진 도커이미지를 다운로드 해서 컨테이너 띄우기

```bash
docker pull ratsgo/embedding-cpu:1.2
docker run -it --rm ratsgo/embedding-cpu:1.2 bash
```