# Just for testing new features of json2csds module.
import json

from json2csds import JSON2CSDS

address = "..\mpqa_dataprocessing\databases\database.mpqa.3.0.cleaned"
obj = JSON2CSDS("MPQA3.0", address)
# Gather the JSON file from MPQA.
mpqa_json = obj.produce_json_file()
json_output = obj.doc2csds(mpqa_json, json_output=True)

# Path is where you want to save the JSON file.
path = ''

with open(path + 'data.json', 'w', encoding='utf-8') as f:
    json.dump(json_output, f, ensure_ascii=False, indent=4)

# Loading the saved JSON file.
with open(path + 'data.json') as json_file:
    data = json.load(json_file)
