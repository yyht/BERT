sudo nvidia-docker run -it -d -p 8538:8888 --name tf_1_15_albert \
--privileged \
-e LANG=C.UTF-8 \
-v /home/albert.xht/source:/notebooks/source \
-v /gruntdata/albert.xht:/data/xuht \
-p 2152:6006 \
-p 6929:8080 \
-p 5339:8891 \
-p 7517:8011 \
-e PASSWORD=123456 \
tensorflow/tensorflow:1.15.0-gpu-py3

sudo docker exec -it tf_1_15_albert bash

