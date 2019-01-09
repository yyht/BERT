sudo nvidia-docker run -it -d --name tensorflow-serving \
-e LANG=C.UTF-8 \
-p 19750:8500 \
-p 19751:8501 \
-v /gruntdata/albert.xht/:/serving \
tensorflow/serving:1.8.0-devel