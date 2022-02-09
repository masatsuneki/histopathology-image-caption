FROM tensorflow/tensorflow:2.5.0-gpu

RUN apt-get update && apt-get -y install git \
  && rm -rf /var/lib/apt/lists/*

WORKDIR app

ADD ./requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

ADD train.py .