FROM tensorflow/tensorflow:1.8.0-py3

RUN apt update && apt install -y vim supervisor git 
Workdir /app
COPY requirements.txt /app
RUN pip install -r requirements.txt
COPY . /app
RUN sh init.sh
RUN  cd  jieba && python3 setup.py install
RUN mkdir -p /var/log/supervisor
COPY config/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

CMD ["/usr/bin/supervisord"]
