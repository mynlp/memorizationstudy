import json
from pythia.utils.mmap_dataset import MMapIndexedDataset
import numpy as np
from tqdm import tqdm
import argparse

def onegram_dict(data):
    fre_dict = {}
    for inst in data:
        tmp_dict = dict(zip(*np.unique(data[inst], return_counts=True)))
        fre_dict = update_dict(fre_dict, tmp_dict)
    return fre_dict

def twogram_dict(data):
    fre_dict = {}
    for inst in data:
        tmp = inst.astype(np.int32)
        targets = []
        if tmp.size % 2 == 0:
            targets.append(tmp)
            targets.append(np.insert(np.insert(tmp, tmp.size, -1), 0, -1))
        else:
            targets.append(np.insert(tmp, 0, -1))
            targets.append(np.insert(tmp, tmp.size, -1))

        for t in targets:
            tr = t.reshape(-1, 2)
            sorted_tr = tr[np.lexsort(tr.T), :]
            diff_idx = np.where(np.any(np.diff(sorted_tr, axis=0), 1))[0]
            unique_rows = [str(sorted_tr[i].tolist()) for i in diff_idx] + [str(sorted_tr[-1].tolist())]
            counts = np.diff(np.append(np.insert(diff_idx, 0, -1), sorted_tr.shape[0] - 1))
            tmp_dict = dict(zip(unique_rows, counts))
            fre_dict = update_dict(fre_dict, tmp_dict)
    return fre_dict

def threegram_dict(src, output):
    bar = tqdm(total = len(src))
    fre_dict = {}
    for inst in src:
        input = inst.astype(np.int32)
        if input.size % 3 == 1:
            input = np.insert(input, input.size, [-1, -1])
        elif input.size % 3 == 2:
            input = np.insert(input, input.size, -1)

        targets = [input]
        targets.append(np.insert(np.insert(input, input.size, [-1, -1]), 0, -1))
        targets.append(np.insert(np.insert(input, input.size, -1), 0, [-1, -1]))

        for t in targets:
            tr = t.reshape(-1, 3)
            sorted_tr = tr[np.lexsort(tr.T), :]
            diff_idx = np.where(np.any(np.diff(sorted_tr, axis=0), 1))[0]
            unique_rows = [str(sorted_tr[i].tolist()) for i in diff_idx] + [str(sorted_tr[-1].tolist())]
            counts = np.diff(np.append(np.insert(diff_idx, 0, -1), sorted_tr.shape[0] - 1)).tolist()
            tmp_dict = dict(zip(unique_rows, counts))
            fre_dict = update_dict(fre_dict, tmp_dict)

        bar.update(1)

    json.dump(fre_dict, open(output, 'w'))
    return 1

def update_dict(dict1, dict2):
    for key in dict2:
        if key == -1 or "-1" in key:
            continue
        if key in dict1:
            dict1[key] = dict1[key] + dict2[key]
        else:
            dict1[key] = dict2[key]
    return dict1

def main(args):
    mmap_ds = MMapIndexedDataset(args.pile_path, skip_warmup=True)
    if args.n == "1":
        ngram_dict = onegram_dict(mmap_ds)
        json.dump(ngram_dict, open(args.o, 'w'))
    elif args.n == "2":
        ngram_dict = twogram_dict(mmap_ds)
        json.dump(ngram_dict, open(args.o, 'w'))
    elif args.n == "3":
        threegram_dict(mmap_ds, args.o)
    else:
        print("this script only supports 1, 2, and 3-gram dictionary")
        return 0

    return 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pile_path', default='./document.bin')
    parser.add_argument('-n', '--number_n_gram', default="1")
    parser.add_argument('-o', '--output', default='result.json')
    args = parser.parse_args()

    main(args)
