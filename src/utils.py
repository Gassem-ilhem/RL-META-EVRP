import io
import json
from os.path import exists

def load_instance(json_file):
    if exists(json_file):
        with io.open(json_file, 'rt', encoding='utf-8', newline='') as file_object:
            instance_data = json.load(file_object)
        return instance_data
    else:
        raise ValueError(f"{json_file} does not exist.")