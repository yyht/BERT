sudo nvidia-docker run -it -d -p 8618:8888 --name tensorflow_1.8_xuht_bert_py3 \
--privileged \
-e LANG=C.UTF-8 \
-v /home/xuht/source:/notebooks/source \
-v /data/xuht:/data/xuht \
-p 6552:6006 \
-p 7999:8080 \
-p 8399:8891 \
-p 8537:8011 \
-e PASSWORD=123456 \
tensorflow/tensorflow:1.8.0-gpu-py3

sudo docker exec -it tensorflow_1.8_xuht_bert_py3 bash
