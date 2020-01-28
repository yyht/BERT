from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import math
import re
import six

import json
from bunch import Bunch

def save_namespace(FLAGS, out_path):
  FLAGS_dict = vars(FLAGS)
  with open(out_path, 'w') as fp:
      #json.dump(FLAGS_dict, fp)
    json.dump(FLAGS_dict, fp, indent=4, sort_keys=True)
        
def load_namespace(in_path):
  with open(in_path, 'r') as fp:
    FLAGS_dict = json.load(fp)
    
    if FLAGS_dict.get("add_position_timing_signal", False) == True:
        FLAGS_dict["pos"] = None
    for key in FLAGS_dict:
        print(key, FLAGS_dict[key])
    return Bunch(FLAGS_dict)

class Config(object):
  """Configuration for `Bert Framework Model`."""

  def __init__(self, config_path):
    self.config = load_namespace(config_path)

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


