# Just for testing new features of json2csds module.
import json

from json2csds import JSON2CSDS

address = "..\mpqa_dataprocessing\database.mpqa.cleaned.221201"
obj = JSON2CSDS("MPQA2.0", address, mpqa_version=2)
# Gather the JSON file from MPQA.
mpqa_json = obj.produce_json_file()
json_output = obj.doc2csds(mpqa_json, json_output=True)

# Path is where you want to save the JSON file.
path = ''

with open(path + 'MPQA.json', 'w', encoding='utf-8') as f:
    json.dump(mpqa_json, f, ensure_ascii=False, indent=4)


with open(path + 'MPQA2.0_v221205_org.json', 'w', encoding='utf-8') as f:
    json.dump(json_output, f, ensure_ascii=False, indent=4)

# Loading the saved JSON file.
with open(path + 'MPQA2.0_v221205_org.json', encoding='utf-8') as json_file:
    data = json.load(json_file)

print(data.keys())