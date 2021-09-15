"""
test file structure
"""
import os
from os.path import join as pj
from utils import readJson, readLines

def test_structure(data, domain, website):
    """
    test the completeness of files
    """
    files = os.listdir(data)
    assert "dataset.csv" in files
    assert "processed_data" in files
    assert len(files) == 2

def test_files(data, domain, website):
    def not_valid(name):
        for t in ['.html', '.png','.txt','.json']:
            if t in name:
                return False
        return True
    # check whether there is any additional files
    files = os.listdir(pj(data, "processed_data"))
    extra_files = [f for f in files if not_valid(f)]
    assert len(extra_files) == 0
    # check four necessary files for each page
    files = [f.split(".")[0] for f in files]
    files = [f for f in files if f!=""]
    for f in files:
        for ftype in ['.html','.json','.txt','.png']:
            assert os.path.exists(pj(data, "processed_data", f+ftype))


def test_json(data, domain, website):
    """
    test the content of meta data file
    """
    files = os.listdir(pj(data, "processed_data"))
    files = [f for f in files if f.endswith(".json")]
    for f in files:
        rect = readJson(pj(data, "processed_data", f))
        assert len(rect) != 0

def test_txt(data, domain, website):
    """
    test the content of text file
    """
    def isUrl(line):
        return line.startswith("http")
    files = os.listdir(pj(data, "processed_data"))
    files = [f for f in files if f.endswith(".txt")]
    for f in files:
        lines = readLines(pj(data, "processed_data", f))
        try:
            assert len(lines) == 2
        except AssertionError:
            print(f)
            raise
        assert isUrl(lines[1]) == True
