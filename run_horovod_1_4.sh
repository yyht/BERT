sudo nvidia-docker run -it -d -p 8418:8888 --name uber_distributed_xuht_py3_tf1_4 \
--privileged \
-e LANG=C.UTF-8 \
-v /home/albert.xht/source:/notebooks/source \
-v /gruntdata/albert.xht:/data/xuht \
-p 6452:6006 \
-p 7549:8080 \
-p 8139:8891 \
-p 8417:8011 \
-e PASSWORD=123456 \
uber/horovod:0.13.8-tf1.8.0-torch0.4.0-py3.5

sudo docker exec -it uber_distributed_xuht_py3_tf1_4 bash
