import json
label_map = {'No_DR': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3, 'Proliferative_DR': 4}
with open("labels.json","w") as f: json.dump(label_map, f)
