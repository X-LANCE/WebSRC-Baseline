"""
test dataset header
"""
import pandas as pd
import os
from os.path import join as pj

def test_header(data, domain, website):
    """
    test the format of dataset.csv
    """
    dataset = pd.read_csv(pj(data, "dataset.csv"))
    assert set(dataset.columns) == set(['question', 'id', 'element_id', 'answer_start', 'answer'])