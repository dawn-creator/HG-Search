from .base_task import BaseTask
TASK_REGISTRY={}
import importlib 

def register_task(name):

    def register_task_cls(cls):
        if name in TASK_REGISTRY:
            raise ValueError("Cannot register duplicate tasks ({})".format(name))
        if not issubclass(cls, BaseTask):
            raise ValueError("Task ({}: {}) must extend BaseTask".format(name, cls.__name__))
        TASK_REGISTRY[name] = cls
        return cls
    
    return register_task_cls 



SUPPORTED_TASKS = {
    'node_classification': 'openhgnn.tasks.node_classification'
}

from .node_classification import NodeClassification


def try_import_task(task):
    if task not in TASK_REGISTRY:
        if task in SUPPORTED_TASKS:
            importlib.import_module(SUPPORTED_TASKS[task])
        else:
            print(f"Failed to import {task} task.")
            return False
    return True


def build_task(args):
    if not try_import_task(args.task):
        exit(1)
    return TASK_REGISTRY[args.task](args)