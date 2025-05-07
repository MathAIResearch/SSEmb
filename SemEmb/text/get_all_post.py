import pandas as pd
from tqdm import tqdm
from lxml import etree
import torch

def estimate_total_lines(xml_file):
    with open(xml_file, 'rb') as f:
        return sum(1 for _ in f) 

def parse_xml_to_dict(xml_file):
    post_dict = {}
    total_lines = estimate_total_lines(xml_file) 
    context = etree.iterparse(xml_file, events=("start",), tag="row")

    for event, elem in tqdm(context, total=total_lines, desc="Parsing XML", unit=" lines"):
        post_id = elem.get("Id")
        if post_id:
            post_dict[post_id] = {key: elem.get(key, "") for key in ["Title", "Body", "Tags"]}
        elem.clear()
    
    return post_dict


xml_file = 'Posts.V1.3.xml'
print("Estimating XML file size...")
post_dict = parse_xml_to_dict(xml_file) 
torch.save(post_dict,'all_post.pt')