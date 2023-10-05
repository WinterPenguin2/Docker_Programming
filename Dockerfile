FROM python:3.8

WORKDIR /app

RUN pip install torch gym==0.24.1 numpy



COPY ddpg_code.py /app/ddpg_code.py


CMD ["python","ddpg_code.py"]
