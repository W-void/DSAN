#!/usr/bin/python
# -*- coding:utf-8 -*-

import traceback

from model.model_lib.EstimatorWithFastEvaluate import EstimatorWithFastEvaluate
from utils.utils import *
from utils.config import TaskConfig
from model.model_lib.estimator import Estimator
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("chief_hosts", "", "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("evaluator_hosts", "", "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", "chief", "One of 'ps', 'worker', 'chief', 'evaluator'")
flags.DEFINE_integer("task_index", 0, "Index of task within the job")

flags.DEFINE_string("task", 'train', "task: train_eval, train, evaluate or infer")
flags.DEFINE_string("model_dir", "s", "model_dir")

flags.DEFINE_string("config_file", "task_conf.json", "config name")
flags.DEFINE_string("data_struct_file", "data_struct.json", "data struct file name")
FLAGS = flags.FLAGS


def main(_):
    logger = set_logger()

    task_config = TaskConfig(FLAGS.config_file, FLAGS.data_struct_file)
    init_environment(FLAGS, task_config.params, logger)

    model_class = import_class(task_config.params['model_class_name'])
    model = model_class(task_config.data_struct,
                        task_config.params,
                        logger,
                        **task_config.params)
    if int(task_config.params['is_dist']) == 0 and int(task_config.params['local_debug']) == 1:
        run_config = init_single_run_config(FLAGS, task_config.params, logger)
    else:
        run_config = init_run_config(FLAGS, task_config.params, logger)
    estimator = EstimatorWithFastEvaluate(model, task_config.data_struct, run_config,
                          logger, FLAGS, task_config.params)

    try:
        if FLAGS.task == 'train':
            estimator.train()
        elif FLAGS.task == 'evaluate':
            estimator.evaluate()
        elif FLAGS.task == 'infer':
            estimator.infer()
        else:
            raise ValueError('Run task not exist!')
    except Exception as e:
        exc_info = traceback.format_exc(sys.exc_info())
        msg = 'creating session exception:%s\n%s' % (e, exc_info)
        tmp = 'Run called even after should_stop requested.'
        should_stop = type(e) == RuntimeError and str(e) == tmp
        if should_stop:
            logger.warn(msg)
        else:
            logger.error(msg)
        # 0 means 'be over', 1 means 'will retry'
        exit_code = 0 if should_stop else 1
        sys.exit(exit_code)


if __name__ == "__main__":
    tf.app.run()
