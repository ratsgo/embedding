---
layout: default
title: 개발환경 설정
description: 도커 설치 및 컨테이너 실행 방법을 안내합니다.
---



## 개발환경 설정

이 페이지에서는 도커 설치 및 컨테이너 실행 방법을 안내합니다.



### 요구사항

(1) 우분투 운영 체제에서 `nvidia-docker` 설치하기 : [링크](https://hiseon.me/2018/02/19/install-docker/)

(2) 윈도우 운영 체제에서 `docker` 설치하기 : [링크](https://steemit.com/kr/@mystarlight/docker)

(3) 윈도우 운영 체제에서 `nvidia-docker` 설치하기 : **nvidia-docker는 윈도우를 지원하지 않습니다([참고](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#is-microsoft-windows-supported))** 아마존 웹 서비스(AWS)를 활용해 개발환경을 구축하는 걸 추천해 드립니다. AWS 사용 방법은 [이곳](https://drive.google.com/open?id=1Zo7_F-ruU5NW9YtAR8IE2zKzm5lz_Cz7)을 참고하세요.



### 도커 실행방법

튜토리얼을 위한 도커 컨테이너를 실행하려면 자신의 환경에 따라 다음 네 가지 중 하나의 작업을 실시하여야 합니다.



#### 로컬에 nvidia-docker가 설치돼 있고 Dockerfile로부터 도커이미지를 처음부터 만들어서 컨테이너 띄우기

```bash
git clone https://github.com/ratsgo/embedding.git
cd embedding
docker build -t ratsgo/embedding-gpu -f docker/Dockerfile-GPU .
docker run -it --rm --gpus all ratsgo/embedding-gpu bash
```



#### 로컬에 nvidia-docker가 설치돼 있고 이미 만들어진 도커이미지를 다운로드 해서 컨테이너 띄우기

```bash
docker pull ratsgo/embedding-gpu
docker run -it --rm --gpus all ratsgo/embedding-gpu bash
```



#### 로컬에 docker가 설치돼 있고 Dockerfile로부터 도커이미지를 처음부터 만들어서 컨테이너 띄우기

```bash
git clone https://github.com/ratsgo/embedding.git
cd embedding
docker build -t ratsgo/embedding-cpu -f docker/Dockerfile-CPU .
docker run -it --rm ratsgo/embedding-cpu bash
```



#### 로컬에 docker가 설치돼 있고 이미 만들어진 도커이미지를 다운로드 해서 컨테이너 띄우기

```bash
docker pull ratsgo/embedding-cpu
docker run -it --rm ratsgo/embedding-cpu bash
```