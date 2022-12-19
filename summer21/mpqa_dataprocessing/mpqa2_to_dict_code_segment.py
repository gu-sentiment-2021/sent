# By running this code segment you'll get a python dictionary or
# a json file that helps you to further examine the MPQA 3.0 dataset
# from mpqa_dataprocessing.mpqa2_to_dict import mpqa2_to_dict
from mpqa2_to_dict import mpqa2_to_dict
import json

# Specifies the path of the mpqa v2.0 database
address = "database.mpqa.cleaned.221201"

# Uses the mpqa2_to_dict module to convert mpqa to json file
m2d = mpqa2_to_dict("MPQA2.0", address)
result = m2d.corpus_to_dict()
# Save the json file to a path
with open("result_mpqa2_cleaned.json", 'w') as file:
    json.dump(result, file, indent=4)

# ##### Find and count OSEs without a sentence
# x, y = 0, 0
# for docname, doc in result['docs'].items():
#     for annoname in doc['direct-subjective']:
#         for anno in doc['annotations'][annoname]:
#             if 'span-in-sentence' not in anno:
#                 x += 1
#                 print(docname, annoname)
#                 # if 'implicit' not in anno:
#                 #     y += 1
# print(x, y)

# ##### Count All annotations
# annos = {}
# for docname, doc in result['docs'].items():
#     for annotype, annonames in doc.items():
#         if annotype != 'annotations':
#             annos[annotype] = annos.get(annotype, 0) + len(annonames)
# print(annos)

# ##### Count number of links in attitudes, DSEs, and ESEs.
# trgt_in_att, src_in_dse, att_in_dse, src_in_ese, src_in_ose = 0, 0, 0, 0, 0
# for docname, doc in result['docs'].items():
#     for annoname in doc['attitude']:
#         for anno in doc['annotations'][annoname]:
#             if 'target-link' in anno:
#                 trgt_in_att += len(anno['target-link'])
#     for annoname in doc['direct-subjective']:
#         for anno in doc['annotations'][annoname]:
#             if 'nested-source' in anno:
#                 src_in_dse += len(anno['nested-source'])
#             if 'attitude-link' in anno:
#                 att_in_dse += len(anno['attitude-link'])
#     for annoname in doc['expressive-subjectivity']:
#         for anno in doc['annotations'][annoname]:
#             if 'nested-source' in anno:
#                 src_in_ese += len(anno['nested-source'])
#     for annoname in doc['objective-speech-event']:
#         for anno in doc['annotations'][annoname]:
#             if 'nested-source' in anno:
#                 src_in_ose += len(anno['nested-source'])
#
# print(trgt_in_att, src_in_dse, att_in_dse, src_in_ese, src_in_ose)
