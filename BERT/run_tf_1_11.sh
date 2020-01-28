sudo nvidia-docker run -it -d -p 8418:8888 --name tensorflow_1.11_xuht_bert_py3 \
--privileged \
-e LANG=C.UTF-8 \
-v /home/xuht/source:/notebooks/source \
-v /data:/data \
-p 6252:6006 \
-p 7199:8080 \
-p 8099:8891 \
-p 8737:8011 \
-e PASSWORD=123456 \
tensorflow/tensorflow:1.11.0-gpu-py3

sudo docker exec -it tensorflow_1.11_xuht_bert_py3 bash
