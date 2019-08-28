import os
import argparse
import pickle
import numpy as np
import json
import py4j
from utilities.common_utils import make_sure_path_exists


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--word2vec', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    df = pickle.load(open(args.word2vec, 'rb'))
    words_map = {v:k for k,v in enumerate(df['word'].values)}
    vectors_matrix = np.array([list(x) for x in df['vector'].values])

    make_sure_path_exists(args.output_dir)

    with open(os.path.join(args.output_dir, 'words_map.json'), 'w') as f:
        json.dump(words_map, f)

    with open(os.path.join(args.output_dir, 'vectors_matrix.pkl'), 'wb') as f:
        pickle.dump(vectors_matrix, f)

if __name__ == '__main__':
    main()