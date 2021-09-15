"""
test answer labels
"""
import pandas as pd
import os
from os.path import join as pj
from bs4 import BeautifulSoup
from utils import readFile, get_node_text, read_dataset
import numpy as np

def test_label(data, domain, website):
    dataset = read_dataset(pj(data, "dataset.csv"))
    files = os.listdir(pj(data, "processed_data"))
    files = [f for f in files if f.endswith(".html")]
    files = {f.split(".")[0]: BeautifulSoup(readFile(pj(data, "processed_data", f)), 'lxml') for f in files}
    for i, line in dataset.iterrows():
        answer = line['answer']
        element_id = line['element_id']
        answer_start = line['answer_start']
        pageid = line['id'][2:9]
        # test yes/no questions
        if element_id == -1 or answer in ['yes', 'no']:
            assert element_id == -1
            assert answer in ['yes', 'no']
            if answer == 'yes':
                assert answer_start == 1
            elif answer == 'no':
                assert answer_start == 0
        else:
            # test the correctness of answer
            page = files[pageid]
            ele = page.find(attrs={'tid':element_id})
            assert ele is not None
            try:
                found_answer = get_node_text(ele)[answer_start:answer_start+len(answer)]
            except TypeError:
                print(line)
                raise
            try:
                assert answer == found_answer
            except AssertionError:
                print(line)
                raise
