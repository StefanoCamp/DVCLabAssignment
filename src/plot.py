import numpy as np
import codecs, json
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import yaml

import constants as c

if len(sys.argv) != 2:
    sys.stderr.write("Arguments error.");
    sys.exit(1)

input = sys.argv[1];

confusion_matrixlist = json.loads(codecs.open(os.getcwd() + "/" + input + 'confusion_matrix.tsv', 'r', encoding='utf-8').read())
confusion_matrix = np.array(confusion_matrixlist);
normalised_confusion_matrixlist = json.loads(codecs.open(os.getcwd() + "/" + input + 'normalised_confusion_matrix.tsv', 'r', encoding='utf-8').read())
normalised_confusion_matrix = np.array(normalised_confusion_matrixlist);

LSTMParams = yaml.safe_load(open('params.yaml'))['LSTM']
n_classes = LSTMParams['n_classes']

# Plot Results:
width = 12
height = 12
plt.figure(figsize=(width, height))
plt.imshow(
    normalised_confusion_matrix,
    interpolation='nearest',
    cmap=plt.cm.rainbow
)
plt.title("Confusion matrix \n(normalised to % of total test data)")
plt.colorbar()
tick_marks = np.arange(n_classes)
plt.xticks(tick_marks, c.LABELS, rotation=90)
plt.yticks(tick_marks, c.LABELS)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
os.makedirs(os.path.join(os.getcwd(),'data', 'plots'), exist_ok=True)
plt.savefig(os.getcwd() + '/data/plots/confusion_matrix.png')