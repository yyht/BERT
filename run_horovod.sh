sudo docker run -it -d -p 8318:8888 --name uber_distributed_xuht_py3 \
--privileged \
-e LANG=C.UTF-8 \
-v /home/albert.xht/source:/notebooks/source \
-v /gruntdata/albert.xht:/data/xuht \
-p 6152:6006 \
-p 7949:8080 \
-p 8339:8891 \
-p 8517:8011 \
-e PASSWORD=123456 \
uber/horovod:0.13.8-tf1.8.0-torch0.4.0-py3.5

sudo docker exec -it uber_distributed_xuht_py3 bash
