"""Helper utilities."""
import pickle
import os
from tqdm import tqdm

class open_w():
    """Open a file for writing (overwrite) with encoding utf-8 in text mode.
        :param filename: Name of file
        :param append: Opens the file for appending if true
        :return: file handle
    """

    def __init__(self, filename, append=False):
        self.filename = filename
        self.append = append
        self.fd = None
    def __enter__(self):
        self.fd = open(self.filename, 'w' if not self.append else 'a', encoding='utf-8')
        return self.fd
    def __exit__(self, type, value, traceback):
        print('Wrote ' + pretty.fname(self.fd.name))
        self.fd.close()

def open_r(filename):
    """Open a file for reading with encoding utf-8 in text mode."""
    return open(filename, 'r', encoding='utf-8')

def do_pickle(obj, filename, message="pickle", protocol=3):
    """Pickle an object and show a message."""
    pretty.start('Dumping ' + message + ' to ' + pretty.fname(filename))
    pickle.dump(obj, open(filename, 'wb'),protocol=protocol)
    pretty.ok()

def dump_text_to_list(filename, dump_path):
    """Dump space separated list of lists from text file to pickle."""
    pretty.start('Dumping ' + pretty.fname(filename) + ' to ' + pretty.fname(dump_path))
    with open(filename, 'r', encoding='utf-8') as edit_file:
        edit_list = [list(map(int, line.split(' '))) for line in edit_file.read().splitlines() if line]
        pickle.dump(edit_list, open(dump_path, "wb"))
        pretty.ok()

def assert_fileexists(filename, info='data'):
    pretty.start('Checking for ' + pretty.fname(filename))
    if not os.path.exists(filename):
        pretty.fail('NOT FOUND')
        pretty.fail('Fatal Error - FILE NOT FOUND')
        exit()
    pretty.ok()

def read_file(filename, info='data'):
    pretty.start('Reading ' + info + ' from ' + pretty.fname(filename))
    if not os.path.exists(filename):
        pretty.fail('NOT FOUND')
        pretty.fail('Fatal Error - FILE NOT FOUND')
        exit()

    with open_r(filename) as file:
        ans = file.read()
        pretty.ok()

    return ans

def read_file_lines(filename, info='data'):
    return read_file(filename, info).splitlines()

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class pretty:
    @staticmethod
    def start(operation):
        print(str(operation) + ' - ', end='', flush=True)

    @staticmethod
    def ok(message='OK'):
        print(bcolors.OKGREEN + str(message) + bcolors.ENDC)

    @staticmethod
    def fail(message="FAIL"):
        print(bcolors.FAIL + str(message) + bcolors.ENDC)

    @staticmethod
    def warn(message="WARNING"):
        print(bcolors.WARNING + str(message) + bcolors.ENDC)

    @staticmethod
    def pheader(message):
        print(bcolors.HEADER + str(message) + bcolors.ENDC)

    @staticmethod
    def fname(message):
        return bcolors.OKBLUE + str(message) + bcolors.ENDC

    @staticmethod
    def passert(condition, message='Test'):
        pretty.start(message)
        if condition:
            pretty.ok()
            return True
        else:
            pretty.fail()
            return False

    @staticmethod
    def assert_gt(a, b, message='Test'):
        """Assert if a is greater than b."""
        return pretty.passert(a > b, str(message) + ' - ' + str(a) + ' > ' + str(b))

    @staticmethod
    def assert_eq(a, b, message='Test'):
        """Assert if a is equal to b."""
        return pretty.passert(a == b, message)

    @staticmethod
    def assert_in(a, b, message='Test'):
        """Assert if a is in b."""
        return pretty.passert(a in b, message) 


def generator_based_read_file(filename, info='data',do_split=False,map_to_int=False):
    batch_size=10000
    #pretty.start('Reading ' + info + ' from ' + pretty.fname(filename))
    if not os.path.exists(filename):
        pretty.fail('NOT FOUND')
        pretty.fail('Fatal Error - FILE NOT FOUND')
        exit()

    with open_r(filename) as file:
        result = []
        for i,line in enumerate(file):
            out = line.strip()
            if do_split:
                out = out.split()
            if map_to_int:
                out = list(map(int,out))
            result.append(out)
            if i and i%(batch_size-1)==0:
                yield result
                result = []
        if len(result)>0:
            yield result
        #pretty.ok()

def read_file_lines(filename, info='data'):
    return read_file(filename, info).splitlines()

def custom_tokenize(sentence, tokenizer, mode="test"):
    #tokenizes the sentences
    #adds [CLS] (start) and [SEP] (end) token
    tokenized = tokenizer.tokenize(sentence,mode)
    tokenized = ["[CLS]"] + tokenized + ["[SEP]"]
    return tokenized