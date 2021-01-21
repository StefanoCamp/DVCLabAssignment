import numpy as np
import codecs, json
import os
import sys
import constants as c

if len(sys.argv) != 2:
    sys.stderr.write("Arguments error.");
    sys.exit(1)

input = sys.argv[1];

DATASET_PATH = os.getcwd() + "/" + input
TRAIN = "train/"
TEST = "test/"

# Load "X" (the neural network's training and testing inputs)

def load_X(X_signals_paths):
    X_signals = []

    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]]
        )
        file.close()

    return np.transpose(np.array(X_signals), (1, 2, 0))

X_train_signals_paths = [
    DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in c.INPUT_SIGNAL_TYPES
]
X_test_signals_paths = [
    DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in c.INPUT_SIGNAL_TYPES
]

X_train = load_X(X_train_signals_paths)
X_test = load_X(X_test_signals_paths)


# Load "y" (the neural network's training and testing outputs)

def load_y(y_path):
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    file.close()

    # Substract 1 to each output class for friendly 0-based indexing
    return y_ - 1

y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
y_test_path = DATASET_PATH + TEST + "y_test.txt"

y_train = load_y(y_train_path)
y_test = load_y(y_test_path)

os.makedirs(os.path.join(os.getcwd(),'data', 'prepared'), exist_ok=True)
json.dump(X_train.tolist(), codecs.open(os.getcwd() + '/data/prepared/X_train.tsv', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
json.dump(X_test.tolist(), codecs.open(os.getcwd() + '/data/prepared/X_test.tsv', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
json.dump(y_train.tolist(), codecs.open(os.getcwd() + '/data/prepared/y_train.tsv', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
json.dump(y_test.tolist(), codecs.open(os.getcwd() + '/data/prepared/y_test.tsv', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

