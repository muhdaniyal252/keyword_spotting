FROM python:3.10.14-bullseye

WORKDIR /workspace

COPY requirements.txt /workspace/

RUN pip install -r requirements.txt

CMD [ "python3", "main.py" ]

