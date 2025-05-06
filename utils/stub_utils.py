# stubs a checkpoint for making things faster
import os 
import pickle


def save_stub(stub_path, object):
    if not os.path.exists(os.path.dirname(stub_path)):
        os.makedirs(os.path.dirname(stub_path))
    if stub_path is not None:
        with open(stub_path, 'wb') as f:
            pickle.dump(object, f)

def read_stub(read_stub_flg, stub_path):
    if read_stub_flg and stub_path is not None and os.path.exists(stub_path):
        with open(stub_path, 'rb') as f:
            object = pickle.load(f)
            return object
    return None