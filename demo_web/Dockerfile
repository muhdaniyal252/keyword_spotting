FROM python:3.10.14-bullseye

COPY requirements.txt /

RUN pip install -r requirements.txt

WORKDIR /shareddrive/working/keyword_spotting/web_app

CMD [ "python3", "main.py" ]

