from .base_flow import BaseFlow

import importlib 



FLOW_REGISTRY = {}
SUPPORTED_FLOWS={
    'node_classification': 'openhgnn.trainerflow.node_classification',
}

def register_flow(name):
    

    def register_flow_cls(cls):
        if name in FLOW_REGISTRY:
            raise ValueError("Cannot register duplicate flow ({})".format(name))
        if not issubclass(cls, BaseFlow):
            raise ValueError("Flow ({}: {}) must extend BaseFlow".format(name, cls.__name__))
        FLOW_REGISTRY[name] = cls
        return cls

    return register_flow_cls

from .node_classification import NodeClassification

def try_import_flow(flow):
    if flow not in FLOW_REGISTRY:
        if flow in SUPPORTED_FLOWS:
            importlib.import_module(SUPPORTED_FLOWS[flow])
        else:
            print(f"Failed to import {flow} flows.")
            return False
    return True


def build_flow(args, flow_name):
    if not try_import_flow(flow_name):
        exit(1)
    return FLOW_REGISTRY[flow_name](args)
