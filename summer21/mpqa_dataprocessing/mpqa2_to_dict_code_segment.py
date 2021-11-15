# By running this code segment you'll get a python dictionary or
# a json file that helps you to further examine the MPQA 3.0 dataset
from mpqa_dataprocessing.mpqa2_to_dict import mpqa2_to_dict
import json

# Specifies the path of the mpqa v2.0 database
address = "database.mpqa.cleaned"

# Uses the mpqa2_to_dict module to convert mpqa to json file
m2d = mpqa2_to_dict("MPQA2.0", address)
result = m2d.corpus_to_dict()
# Save the json file to a path
with open("result_mpqa2.json", 'w') as file:
    json.dump(result, file, indent=4)
