sudo nvidia-docker run -it -d \
--privileged \
--user root \
--name uber_distributed_xuht_py3 \
--network=host \
-e LANG=C.UTF-8 \
-v /home/xuht/source:/notebooks/source \
-v /data/xuht:/data/xuht \
-v /mnt:/mnt \
-v /home/xuht/source/key:/root/.ssh \
-e PASSWORD=123456 \
uber/horovod:0.13.8-tf1.8.0-torch0.4.0-py3.5 \
bash -c "/usr/sbin/sshd -p 12345; sleep infinity"
