import numpy as np
import codecs, json
import os
import sys
from sklearn import metrics

if len(sys.argv) != 3:
    sys.stderr.write("Arguments error.");
    sys.exit(1)

input = sys.argv[1];
input2 = sys.argv[2];

accuracylist = json.loads(codecs.open(os.getcwd() + "/" + input + 'acc.tsv', 'r', encoding='utf-8').read())
accuracy = np.array(accuracylist);
ohplist = json.loads(codecs.open(os.getcwd() + "/" + input + 'ohp.tsv', 'r', encoding='utf-8').read())
one_hot_predictions = np.array(ohplist);
y_testlist = json.loads(codecs.open(os.getcwd() + "/" + input2 + 'y_test.tsv', 'r', encoding='utf-8').read())
y_test = np.array(y_testlist);

# Results

predictions = one_hot_predictions.argmax(1)

print("Testing Accuracy: {}%".format(100*accuracy))

print("")
print("Precision: {}%".format(100*metrics.precision_score(y_test, predictions, average="weighted")))
print("Recall: {}%".format(100*metrics.recall_score(y_test, predictions, average="weighted")))
print("f1_score: {}%".format(100*metrics.f1_score(y_test, predictions, average="weighted")))

print("")
print("Confusion Matrix:")
confusion_matrix = metrics.confusion_matrix(y_test, predictions)
print(confusion_matrix)
normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100

print("")
print("Confusion matrix (normalised to % of total test data):")
print(normalised_confusion_matrix)
print("Note: training and testing data is not equally distributed amongst classes, ")
print("so it is normal that more than a 6th of the data is correctly classifier in the last category.")

os.makedirs(os.path.join(os.getcwd(),'data', 'results'), exist_ok=True)
json.dump(confusion_matrix.tolist(), codecs.open(os.getcwd() + '/data/results/confusion_matrix.tsv', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
json.dump(normalised_confusion_matrix.tolist(), codecs.open(os.getcwd() + '/data/results/normalised_confusion_matrix.tsv', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

