import json
import pandas as pd
import os
from bs4.element import NavigableString, Tag

def readFile(path):
    with open(path, "r") as f:
        return f.read()

def readLines(path):
    with open(path, "r") as f:
        return f.readlines()


def readJson(path):
    with open(path, "r") as f:
        return json.loads(f.read())

def read_dataset(path):
    dataset = pd.read_csv(path, dtype={"pageid":str, "answer":str})
    return dataset

def get_node_text(node):
    """
    return the text of a node
    """
    text = []
    for child in node.contents:
        if type(child) == NavigableString:
            if child.strip()!="":
                text.append(child.strip())
        if type(child) == Tag:
            t = get_node_text(child)
            if t!="":
                text.append(t)
    return " ".join(text)

def get_domain(path):
    domains = os.listdir(path)
    registered_domain = ['auto','book','camera','jobs','restaurant','sports','movie','university','game','hotel','computer','phone']
    domains = [d for d in domains if d in registered_domain]
    return domains