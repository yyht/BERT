from __future__ import print_function

import argparse
import time
import numpy as np
import json

from grpc.beta import implementations
from tensorflow.contrib.util import make_tensor_proto

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

def prepare_grpc_request(model_name, signature_name, input_dict):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.signature_name = signature_name
    for key in input_dict:
        request.inputs[key].CopyFrom(
            make_tensor_proto(input_dict[key], dtype=None))
    return request

def run(host, port, test_json, model_name, signature_name):

    # channel = grpc.insecure_channel('%s:%d' % (host, port))
    channel = implementations.insecure_channel(host, port)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    with open(test_json, "r") as frobj:
        content = json.load(frobj)

    start = time.time()

    for input_dict in content:
        request = prepare_grpc_request(model_name, signature_name, input_dict)
        result = stub.Predict(request, 10.0)
        print(result)

    end = time.time()
    time_diff = end - start
    print('time elapased: {}'.format(time_diff))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', help='Tensorflow server host name', default='localhost', type=str)
    parser.add_argument('--port', help='Tensorflow server port number', default=8500, type=int)
    parser.add_argument('--data', help='input image', type=str)
    parser.add_argument('--model_name', help='model name', type=str)
    parser.add_argument('--signature_name', help='Signature name of saved TF model',
                        default='serving_default', type=str)

    args = parser.parse_args()
    run(args.host, args.port, args.data, args.model_name, args.signature_name)