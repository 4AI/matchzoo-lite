FROM python:3.7.2
LABEL maintainer="Sean Lee <xmlee97@gmail.com>"

USER root
ENV USER root

RUN git clone https://github.com/4AI/matchzoo-lite.git &&\ 
    cd matchzoo-lite &&\ 
    python setup.py install &&\ 
    pip install pytest-cov &&\ 
    pytest --cov tests
