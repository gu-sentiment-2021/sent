from mpqa_dataprocessing.mpqa3_to_dict import mpqa3_to_dict
import json

# Specifies the path of the mpqa v3.0 database
address = "..\mpqa_dataprocessing\databases\database.mpqa.3.0.cleaned"

# Uses the mpqa3_to_dict module to convert mpqa to json file
m2d = mpqa3_to_dict("MPQA3.0", address)
result = m2d.corpus_to_dict()
# Save the json file to a path
with open("result_mpqa3.json", 'w') as file:
    json.dump(result, file, indent=4)
