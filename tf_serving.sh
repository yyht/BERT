sudo nvidia-docker run -it -d --name tensorflow-serving \
-e LANG=C.UTF-8 \
-p 7900:8500 \
-p 7901:8501 \
-v /gruntdata/albert.xht/:/serving \
tensorflow/serving:1.8.0-devel
