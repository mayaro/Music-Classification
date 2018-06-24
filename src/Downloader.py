import requests
import multiprocessing as mp
from math import ceil
from pytube import YouTube

CHUNK_SIZE = 1 * 2**20

def download(stream, filename):
    ranges = [[stream.url, i * CHUNK_SIZE, (i+1) * CHUNK_SIZE - 1] for i in range(ceil(stream.filesize / CHUNK_SIZE))]
    ranges[-1][2] = None

    pool = mp.Pool(min(len(ranges), 32))
    chunks = pool.map(__download_chunk, ranges)

    with open(filename, 'wb') as outfile:
        for chunk in chunks:
            outfile.write(chunk)

def __download_chunk(args):
    url, start, finish = args
    range_string = '{}-'.format(start)

    if finish is not None:
        range_string += str(finish)

    response = requests.get(url, headers={'Range': 'bytes=' + range_string})
    return response.content