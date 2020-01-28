docker exec -it event mkdir -p /data/xuht/bert/chinese_L-12_H-768_A-12
docker cp /data/xuht/bert/chinese_L-12_H-768_A-12 event:/data/xuht/bert
docker exec -it event mkdir -p /data/xuht/eventy_detection/event/model
docker cp /data/xuht/eventy_detection/event/model event:/data/xuht/eventy_detection/event

