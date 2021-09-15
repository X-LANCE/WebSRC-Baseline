def pytest_addoption(parser):
    parser.addoption("--data", action="store", type=str, default="../data/auto/03",
        help="path to dataset")
    parser.addoption("--domain", action="store",type=str, default="auto",
        help="domain to test")
    parser.addoption("--website", action="store",type=str, default="website",
        help="website to test")

def pytest_generate_tests(metafunc):
    if "data" in metafunc.fixturenames:
        metafunc.parametrize("data", [metafunc.config.getoption("data")])
    if "domain" in metafunc.fixturenames:
        metafunc.parametrize("domain", [metafunc.config.getoption("domain")])
    if "website" in metafunc.fixturenames:
        metafunc.parametrize("website", [metafunc.config.getoption("website")])