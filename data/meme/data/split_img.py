import json
from shutil import copyfile
from tqdm import tqdm

with open('./train.jsonl', 'r') as json_file:
    json_list = list(json_file)

for json_str in tqdm(json_list):
    result = json.loads(json_str)
    img_path = result['img']
    copyfile(img_path, '../train2014/' + img_path.split('/')[-1])

with open('./test.jsonl', 'r') as json_file:
    json_list = list(json_file)

for json_str in tqdm(json_list):
    result = json.loads(json_str)
    img_path = result['img']
    copyfile(img_path, '../val2014/' + img_path.split('/')[-1])