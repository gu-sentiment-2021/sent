# Just for testing new features of json2csds module.
import json

# Path is where you want to save the JSON file.
path = ''


# Loading the saved JSON file.
with open(path + 'MPQA2.0_v221205_cleaned.json', encoding='utf-8') as json_file:
    data = json.load(json_file)

ese = 0
for item in data['csds_objects']:
    if item['annotation_type'] == 'expressive_subjectivity':
        ese += 1

print(ese)

dse = 0
for item in data['csds_objects']:
    if item['annotation_type'] == 'direct_subjective':
        dse += 1

print(dse)


#attitude?

print(data.keys())