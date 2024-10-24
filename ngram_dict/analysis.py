import argparse
from itertools import product
import json
import logging
from logging import getLogger

from pythia.utils.mmap_dataset import MMapIndexedDataset
from tqdm import tqdm

handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
logger = getLogger(__name__)
handler.setFormatter(formatter)
logger.addHandler(handler)

targets = ["memorized", "forgotten", "half", "quarter"]
sizes = ["70m", "160m", "410m", "1b", "2.8b", "6.9b", "12b"]
context_length = ["32"]
target_length = ["48"]

def read_file(f):
    for line in open(f):
        yield line

def read_mmap_ds(mmap_ds):
    for m in mmap_ds:
        yield m

def make_full_index(result_folder, output):
    index_set = set()
    base_filename = result_folder + "/memorization_evals_{}-deduped-v0_{}_{}_143000.csv"
    for s in sizes:
        fn = base_filename.format(s, context_length[0], target_length[0])
        bar = tqdm(total = 146432001)
        for line in read_file(fn):
            _, index, score = line.strip().split(',')
            if score in ["1.0", "0.5", "0.25", "0.0"]:
                index_set.add(index)
            bar.update(1)

    json.dump(list(index_set), open(output, 'w'))

def make_full_index_dict(dict_path, mmap_path, index_path, output, ngram=1):
    fre_dict = json.load(open(dict_path))
    mmap_ds = MMapIndexedDataset(mmap_path, skip_warmup=True)
    index_keys = json.load(open(index_path))
    index_freq = dict()
    fw = open(output, 'w')

    bar = tqdm(total = len(index_keys))
    for i in index_keys:
        index_freq[i] = []
        m = mmap_ds[int(i)][:47+ngram].tolist()
        for j in range(47+ngram):
            if ngram == 1:
                target = m[j]
            if ngram == 2:
                target = [m[j], m[j+1]]
            elif ngram == 3:
                target = [m[j], m[j+1], m[j+2]]
            else:
                print('we only support 1, 2, and 3-gram')
                return 0
            index_freq[i].append(fre_dict[str(target)])
        fw.write(json.dumps(index_freq) + '\n')
        bar.update(1)

    fw.close()

def write_step_stats(index_path, result_folder, output_folder):
    base_filename = result_folder + "/memorization_evals_{}-deduped-v0_{}_{}_143000.csv"
    for s in sizes:
        memorized, forgotten, half, quarter = set(), set(), set(), set()
        fn = base_filename.format(s, context_length[0], target_length[0])
        bar = tqdm(total = 146432001, desc="loading memorized/forgotten/half/quarter {} ...".format(s))
        for line in open(fn):
            _, index, score = line.strip().split(',')
            if score == "1.0":
                memorized.add(index)
            elif score == "0.0":
                forgotten.add(index)
            elif score == "0.5":
                half.add(index)
            elif score == "0.25":
                quarter.add(index)
            bar.update(1)
        
        bar = tqdm(total = len(memorized), desc="writing memorized {} ...".format(s))
        fn = open(output_folder + '/memorized_{}.tsv'.format(s), 'w')
        for line in open(index_path):
            key = line.split(':')[0][2:-1]
            if key in memorized:
                fn.write('{}\t{}\n'.format(key, line.split(':')[1].strip()[:-1]))
                bar.update(1)
        fn.close()

        bar = tqdm(total = len(forgotten), desc="writing forgotten {} ...".format(s))
        fn = open(output_folder + '/forgotten_{}.tsv'.format(s), 'w')
        for line in open(index_path):
            key = line.split(':')[0][2:-1]
            if key in forgotten:
                fn.write('{}\t{}\n'.format(key, line.split(':')[1].strip()[:-1]))
                bar.update(1)
        fn.close()

        bar = tqdm(total = len(half), desc="writing half {} ...".format(s))
        fn = open(output_folder + '/half_{}.tsv'.format(s), 'w')
        for line in open(index_path):
            key = line.split(':')[0][2:-1]
            if key in half:
                fn.write('{}\t{}\n'.format(key, line.split(':')[1].strip()[:-1]))
                bar.update(1)
        fn.close()

        bar = tqdm(total = len(quarter), desc="writing quarter {} ...".format(s))
        fn = open(output_folder + '/quarter_{}.tsv'.format(s), 'w')
        for line in open(index_path):
            key = line.split(':')[0][2:-1]
            if key in quarter:
                fn.write('{}\t{}\n'.format(key, line.split(':')[1].strip()[:-1]))
                bar.update(1)
        fn.close()

def write_whole_stat(step_folder, output):
    fw = open(output, 'w')
    rg = range(48)
    rg = [str(r) for r in rg]
    fw.write('target\tsize\t{}\n'.format('\t'.join(rg)))

    for t, s in product(targets, sizes):
        cnt = 0
        steps = []
        logger.warning('{} {} doing ...'.format(t, s))
        for line in open(step_folder + '/{}_{}.tsv'.format(t, s)):
            line = line.strip()
            if not line:
                continue
            freq_list = eval(line.split('\t')[1])
            for i in range(48):
                if cnt == 0:
                    steps.append(freq_list[i])
                else:
                    steps[i] += freq_list[i]
            cnt += 1
        
        avrs = []
        for i in range(48):
            avrs.append(str(steps[i]/cnt))
        
        fw.write('{}\t{}\t{}\n'.format(t, s, '\t'.join(avrs)))
        logger.warning('{} {} complete ...'.format(t, s))

    fw.close()

def main(args):
    make_full_index(args.result_folder, args.output_file)
    make_full_index_dict(args.dict_path, args.mmap_path, args.index_path, args.output_file, int(args.ngram))
    write_step_stats(args.index_path, args.result_folder, args.output_folder)
    write_whole_stat(args.step_folder, args.output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--result_folder')
    parser.add_argument('--output_file')
    parser.add_argument('-d', '--dict_path')
    parser.add_argument('-m', '--mmap_path')
    parser.add_argument('-i', '--index_path')
    parser.add_argument('-n', '--ngram')
    parser.add_argument('--output_folder')
    parser.add_argument('--step_folder')
    args = parser.parse_args()

    main(args)



