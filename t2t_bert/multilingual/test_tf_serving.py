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

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "vocab", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "url", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "port", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "model_name", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "signature_name", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "input_keys", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

tokenizer = tokenization.FullTokenizer(
				vocab_file=FLAGS.vocab, 
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

def get_single_features(query, max_seq_length):
    tokens_a = tokenizer.tokenize(query)

    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)

    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    label_ids = 0

	return {"input_ids":input_ids,
			"input_mask":input_mask,
			"segment_ids":segment_ids,
			"label_ids":[0]}

def main():

	query_lst = [' เคยกิน แต่ รุ่น เก่า แต่ รุ่น ใหม่ นี้ ต้อง ลองเดี๋ยว ดูก่อนเดี๋ยว มาบอก',
			 ' oakdeals heart crystal zircon i like it very much though the delivery was quite long to wait for but the waiting is worthy it looks expensive particularly the stone just one thing hope it wont get faded quickly',
			 ' ได้รับ แล้ว แต่ แกะ มาเปิด ไม่ ติด แบตเตอรี่ บวม ได้รับ สินค้า แล้ว ครับ แต่ แกะ มา แบตบวม แถม ลอง เอามาชาร์จ ก็เปิด ไม่ ติด เตรียม ส่งคืน ครับ',
			 ' color of the pants the color so different compared to the one they show in the picture',
			 ' ayus naman ok naman di kalakasan pero ayus lang kasi 16 ＼ lang at buy 1 take 1 pa',
			 ' very satisfied it s exactly as per shown and looking pretty good do nt know how long it will last but for that price it s well worth it will order more',
			 ' kurang ajib yang 30ml jeruk nya mantap tapi gak deh buat yang mocha latte 60ml nya perlu riset mendalam soal rasa kopi nya',
			 ' giao sai hàng mua cái tẩu tin tưởng ko mở ra kiểm tra shop gửi cho cái loa usb ghẻ to vãi haizz chả nhẽ 65k vác ra kiểm tra rồi đi đổi',
			 ' poor quality got mine one month ago in less than a week left side always had problem charging remained low battery no matter how long it s charged now its totally dead useless overpriced earphone stay away from this brand',
			 ' gak puas sama jas kurir ninja cukinnya udh dateng baru kali ini di antar ninja expres gak gak lagi lah pake jasa kurir ninja kurirnya gak jelas marah2 nanya alamat wong lagi di jelasan ga mau nyimak dulu bikin emosi aja mending pk kurir mas lazada langsung deh lebih sopan lebih ramah toling di perbaiki jasa kirim ninja']

	features = []

	for query in query_lst:
		feature = get_single_features(query, 128)
		features.append(feature)

	if FLAGS.input_keys == "instances":
		for key in features[0]:
			import numpy as np
			print(np.array(features[0][key]).shape, key)
		feed_dict = {
			"instances":features[0:5],
			"signature_name":FLAGS.signature_name
		}

	elif FLAGS.input_keys == "inputs":
		feed_dict = {
			"inputs":{
				"input_ids":[],
				"input_mask":[],
				"segment_ids":[],
				"label_ids":[]
			},
			"signature_name":FLAGS.signature_name
		}
		for feature in features[0:5]:
			for key in feed_dict["inputs"]:
				if key not in ["label_ids"]:
					feed_dict["inputs"][key].append(feature[key])
				else:
					feed_dict["inputs"][key].extend(feature[key])

		for key in feed_dict["inputs"]:
			print(key, np.array(feed_dict["inputs"][key]).shape)

	results = requests.post("http://%s:%s/v1/models/%s:predict" % (FLAGS.url, FLAGS.port, FLAGS.model_name), json=feed_dict)
	try:
		print(results.json())
	except:
		import json
		print(results.content)

if __name__ == "__main__":
	main()



