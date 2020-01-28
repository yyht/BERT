import argparse
import imghdr
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib.saved_model.python.saved_model import reader
from optparse import OptionParser
import tensorflow.python.tools.freeze_graph

_GRAPH_FILE = "frozen_graph.pb"


def write_graph_to_file(graph_name, graph_def, output_dir):
    """Write Frozen Graph file to disk."""
    output_path = os.path.join(output_dir, graph_name)
    with tf.gfile.GFile(output_path, "wb") as f:
        f.write(graph_def.SerializeToString())


################################################################################
# Utils for handling Frozen Graphs.
################################################################################
def get_serving_meta_graph_def(savedmodel_dir):
    """Extract the SERVING MetaGraphDef from a SavedModel directory.
    Args:
      savedmodel_dir: the string path to the directory containing the .pb
        and variables for a SavedModel. This is equivalent to the subdirectory
        that is created under the directory specified by --export_dir when
        running an Official Model.
    Returns:
      MetaGraphDef that should be used for tag_constants.SERVING mode.
    Raises:
      ValueError: if a MetaGraphDef matching tag_constants.SERVING is not found.
    """
    # We only care about the serving graph def
    tag_set = set([tf.saved_model.tag_constants.SERVING])
    serving_graph_def = None
    saved_model = reader.read_saved_model(savedmodel_dir)
    for meta_graph_def in saved_model.meta_graphs:
        if set(meta_graph_def.meta_info_def.tags) == tag_set:
            serving_graph_def = meta_graph_def
    if not serving_graph_def:
        raise ValueError("No MetaGraphDef found for tag_constants.SERVING. "
                         "Please make sure the SavedModel includes a SERVING def.")

    return serving_graph_def


def convert_savedmodel_to_frozen_graph(savedmodel_dir, output_dir, signature_def_name):
    """Convert a SavedModel to a Frozen Graph.
    A SavedModel includes a `variables` directory with variable values,
    and a specification of the MetaGraph in a ProtoBuffer file. A Frozen Graph
    takes the variable values and inserts them into the graph, such that the
    SavedModel is all bundled into a single file. TensorRT and TFLite both
    leverage Frozen Graphs. Here, we provide a simple utility for converting
    a SavedModel into a frozen graph for use with these other tools.
    Args:
      savedmodel_dir: the string path to the directory containing the .pb
        and variables for a SavedModel. This is equivalent to the subdirectory
        that is created under the directory specified by --export_dir when
        running an Official Model.
      output_dir: string representing path to the output directory for saving
        the frozen graph.
    Returns:
      Frozen Graph definition for use.
    """
    meta_graph_def = get_serving_meta_graph_def(savedmodel_dir)
    signature_def = meta_graph_def.signature_def[signature_def_name]

    print("{}".format([v for v in meta_graph_def.signature_def.keys()]))
    try:
        outputs = [v.name for v in signature_def.outputs.itervalues()]
    except:
        outputs = [v.name for k, v in signature_def.outputs.items()]
    print("output:".format(outputs))
    output_names = [str(node).split(":")[0] for node in outputs]
    print("{}".format(signature_def))

    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        tf.saved_model.loader.load(
            sess, meta_graph_def.meta_info_def.tags, savedmodel_dir)
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, graph.as_graph_def(), output_names)
        #[print(n.name) for n in frozen_graph_def.node]
    write_graph_to_file(_GRAPH_FILE, frozen_graph_def, output_dir)

    return frozen_graph_def


optParser = OptionParser()
optParser.add_option('-i','--saved_model_dir', action='store', type="string", dest='saved_model_dir')
optParser.add_option("-o", "--output_dir", action="store", dest="output_dir",)
optParser.add_option("-s", "--signature_def", action="store", dest="signature_def", default="predict_images")
option, args = optParser.parse_args()

"""
saved_model_dir： savedModel的目录
output_dir：这个输出的模型文件目录
signature_def：这个是保存saved_model的时候传的signature_def
"""

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    convert_savedmodel_to_frozen_graph(option.saved_model_dir, option.output_dir, option.signature_def)