mpirun -np 4 \
 -H 192.168.3.133:4 \
 -mca btl_tcp_if_exclude docker0,lo,veth0f552c8,vetha000527,veth0008561,veth0d11910,vetha41034f,vethae22aec,vethab0ce88,vethfbea832,b,br-1ca618956367,vethb4b2f97,vethafec1f2,br-224fa76e2bea,br-6655561aac78,br-697da8ae7b12,br-3e45e7ddc33c,br-be5c07c9e794,docker_gwbridge,vboxnet0,virbr0 \
 -x NCCL_SOCKET_IFNAME=eno1 \
 -H localhost:4 \
python train_sess.py \
 --config_file "/data/xuht/bert/chinese_L-12_H-768_A-12/bert_config.json" \
 --init_checkpoint "/data/xuht/bert/chinese_L-12_H-768_A-12/bert_model.ckpt" \
 --vocab_file "/data/xuht/bert/chinese_L-12_H-768_A-12/vocab.txt" \
 --label_id "/data/xuht/wsdm19/data/label_dict.json" \
 --train_file "/data/xuht/wsdm19/postprocess/data_12_6/pretrain/train" \
 --dev_file "/data/xuht/wsdm19/postprocess/data_12_6/pretrain/test" \
 --max_length 128 \
 --model_output "/data/xuht/wsdm19/postprocess/data_12_6/pretrain/model_12_11" \
 --epoch 8 \
 --num_classes 3 \
 --batch_size 16 \
 --if_shard "1" \
 --max_predictions_per_seq 5

