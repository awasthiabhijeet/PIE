"""Synthetic data generation by introducing errors."""
import sys
import multiprocessing as mp
from filelock import FileLock
from tqdm import tqdm
from errorifier import Errorifier

CORRECT_FILE = sys.argv[2] + "/" + 'corr_sentences.txt'
ERRORED_FILE = sys.argv[2] + "/" + 'incorr_sentences.txt'
FLUSH_SIZE = 100000
BATCH_SIZE = 200

def flush_queue(pairs, flush=False):
    """Write queue obects to file."""

    wrote = 0
    with FileLock('%s.lock' % CORRECT_FILE), FileLock('%s.lock' % ERRORED_FILE):
        if pairs.qsize() < FLUSH_SIZE and not flush:
            return

        with open(CORRECT_FILE, 'a') as cfile, open(ERRORED_FILE, 'a') as efile:
            while pairs.qsize() > 0 and (wrote < FLUSH_SIZE or flush):
                wrote += 1
                pair = pairs.get()
                cfile.write(pair[0] + '\n')
                efile.write(pair[1] + '\n')

def errorify(tpl):
    """Function to use for multiprocessing."""
    # Unpack
    sentences, pairs = tpl[0], tpl[1]

    for sentence in sentences:
        eff = Errorifier(sentence)
        puttpl = (eff.correct(), eff.error())
        pairs.put(puttpl)

    if pairs.qsize() > FLUSH_SIZE:
        flush_queue(pairs)

def readn(file, n):
    """Read a file n lines at a time."""
    start = True
    clist = []
    for line in file:
        if start:
            clist = []
            start = False
        clist.append(line)
        if len(clist) == n:
            start = True
            yield clist
    yield clist


def errorify_file(filename: str):
    """Errorify all sentences in a file."""

    # Blank files
    open(CORRECT_FILE, 'w').close()
    open(ERRORED_FILE, 'w').close()

    # Threads = CPU count
    pool = mp.Pool(mp.cpu_count())
    manager = mp.Manager()
    pairs = manager.Queue()

    # Erroriy each line
    file = open(filename, 'r')
    [x for x in tqdm(pool.imap(errorify, [(l, pairs) for l in readn(file, BATCH_SIZE)]))]
    pool.close()

    # Flush anything remaining
    flush_queue(pairs, True)

if __name__ == '__main__':
    errorify_file(sys.argv[1])
