"""
test qid and id
"""

import pandas as pd
import os
from os.path import join as pj

def test_id(data, domain, website):
    dataset = pd.read_csv(pj(data, "dataset.csv"))
    domain_short = domain[0:2]
    """
    test the format of id
    """
    for id_ in dataset['id'].values:
        assert len(id_) == 14
        assert id_[0:2] == domain_short
        assert id_[2:4] == website
