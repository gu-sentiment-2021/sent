# Just for testing new features of json2csds module.
import json

# Path is where you want to save the JSON file.
path = ''


# Loading the saved JSON file.
with open(path + 'MPQA2.0_v221205_cleaned.json', encoding='utf-8') as json_file:
    data = json.load(json_file)

ese = 0
agnt_in_ese = 0

dse = 0
agnt_in_dse = 0
att_in_dse = 0

att = 0
trgt_in_att = 0

agnt = len(data['agent_objects'])
trgt = len(data['target_objects'])
sentences = 0
ose = 0

for item in data['csds_objects']:
    if item['annotation_type'] == 'expressive_subjectivity':
        ese += 1
        agnt_in_ese += len(item['nested_source'])
    if item['annotation_type'] == 'direct_subjective':
        dse += 1
        agnt_in_dse += len(item['nested_source'])
        att_in_dse += len(item['attitude'])
    if item['annotation_type'] in ['agreement', 'arguing', 'intention', 'other_attitude', 'sentiment', 'speculation']:
        att += 1
        trgt_in_att += len(item['target'])
    if item['annotation_type'] == 'sentence':
        sentences += 1
    if item['annotation_type'] == 'objective_speech_event':
        ose += 1

print('Agents:', agnt)
print('Targets:', trgt)
print()

print("ESEs:", ese)
print("Agents in ESEs:", agnt_in_ese)
print()

print("DSE:", dse)
print("Attitudes in DSE:", att_in_dse)
print("Agents in DSE:", agnt_in_dse)
print()

print("ATT:", att)
print("Targets in ATTs:", trgt_in_att)
print()

print("OSE:", ose)
print("Sentences:", sentences)
print()

print(data.keys())