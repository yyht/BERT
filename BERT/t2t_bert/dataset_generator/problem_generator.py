from dataset_generator.data_reader import SentenceProcessor, SentencePairProcessor
from data_generator import tokenization
from dataset_generator.create_cls_problem_generator import create_cls_problem_generator
from dataset_generator.create_masked_lm_generator import create_instances_from_document
from dataset_generator.create_pretrain_generator import create_pretraining_generator
from dataset_generator.dataset_utils import _create_dummpy_label
import random

def problem_generator(task_type, examples, label2id, 
				multi_task_config, tokenizer, mode):
	if multi_task_config[task_type]["task_type"] == "cls_task":
		random.shuffle(examples)
		return create_cls_problem_generator(task_type,
                                	examples,
                                	label2id,
                                	multi_task_config,
                                	tokenizer,
                                	mode)