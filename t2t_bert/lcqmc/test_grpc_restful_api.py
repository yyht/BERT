import sys,os
sys.path.append("..")
import numpy as np
import tensorflow as tf
from bunch import Bunch
from data_generator import tokenization
from data_generator import tf_data_utils
from model_io import model_io
import json
import requests
import pickle
import numpy as np
import time
import requests
import subprocess
import re
from grpc.beta import implementations
# from tensorflow_serving.apis import predict_pb2
# from tensorflow_serving.apis import prediction_service_pb2

tokenizer = tokenization.FullTokenizer(
        vocab_file="/data/xuht/chinese_L-12_H-768_A-12/vocab.txt", 
        do_lower_case=True)

def full2half(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code==0x3000:
            inside_code=0x0020
        else:
            inside_code-=0xfee0
        if inside_code<0x0020 or inside_code>0x7e:   
            rstring += uchar
        else:
            rstring += unichr(inside_code)
    return rstring

def get_single_features(query, sent, max_seq_length):
  query = full2half(query)
  tokens_a = tokenizer.tokenize(query)

  sent = full2half(sent)
  tokens_b = tokenizer.tokenize(sent)

  def get_input(input_tokens_a, input_tokens_b):
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)

    for token in input_tokens_a:
      tokens.append(token)
      segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    for token in input_tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    return [tokens, input_ids, 
        input_mask, segment_ids]

  [tokens_a_,
  input_ids_a, 
  input_mask_a, 
  segment_ids_a] = get_input(tokens_a, tokens_b)

  [tokens_b_,
  input_ids_b, 
  input_mask_b, 
  segment_ids_b] = get_input(tokens_b, tokens_a)

  return {"input_ids_a":input_ids_a,
      "input_mask_a":input_mask_a,
      "segment_ids_a":segment_ids_a,
      "input_ids_b":input_ids_b,
      "input_mask_b":input_mask_b,
      "segment_ids_b":segment_ids_b,
      "label_ids":[0]}

# def prepare_grpc_request(model_name, signature_name, data):
#     request = predict_pb2.PredictRequest()
#     request.model_spec.name = model_name
#     request.model_spec.signature_name = signature_name
#     for key in data:
#       request.inputs[key].CopyFrom(
#         tf.contrib.util.make_tensor_proto(data[key], dtype=None))
#     return request

query = u"银行转证券怎么转"
candidate_lst = 10*[u"银行转证券怎么才能转过去"]

features = []

for candidate in candidate_lst:
  feature = get_single_features(query, candidate, 500)
  features.append(feature)

feed_dict = {
      "input_ids_a":[],
      "input_mask_a":[],
      "segment_ids_a":[],
      "input_ids_b":[],
      "input_mask_b":[],
      "segment_ids_b":[],
      "label_ids":[]
 
}

for feature in features:
    for key in feed_dict:
      feed_dict[key].append(feature[key])

host = '10.183.20.12'
grpc_port = '7900'
rest_port = '7901'
model_name = 'default'
signature_name = 'output'

# channel = implementations.insecure_channel(host, int(grpc_port))
# stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

# # gRPC
# print("starting gRPC test...")
# print ("warming up....")
# request = prepare_grpc_request(model_name, signature_name, feed_dict)
# stub.Predict(request, timeout=600)

# REST
print("starting REST test...")
json = {
    "signature_name": signature_name,
    "instances": features[0:2]
}
print ("warming up....")
req = requests.post("http://%s:%s/v1/models/default:predict" % (host, rest_port), json=json)
print(req.json())
