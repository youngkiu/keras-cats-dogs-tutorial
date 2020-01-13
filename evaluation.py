import glob
import os

import keras
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

TEST_DATASET_DIR_PATH = './catsdogs/sample/valid'
INPUT_SHAPE = (224, 224)
BATCH_SIZE = 8

model = keras.models.load_model('model-resnet50-final.h5')
model.summary()


test_datagen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=keras.applications.resnet50.preprocess_input
)

test_generator = test_datagen.flow_from_directory(
    TEST_DATASET_DIR_PATH,
    target_size=INPUT_SHAPE,
    interpolation='bicubic',
    class_mode='binary',
    batch_size=BATCH_SIZE,
)

loss, accuracy = model.evaluate_generator(
    test_generator, steps=len(test_generator), verbose=1)
print('Restored model, accuracy: {:5.2f}%'.format(100*accuracy))

predictions = model.predict_generator(
    test_generator, steps=len(test_generator), verbose=1)

y_gt_classes = test_generator.classes

y_pred_proba = predictions.flatten()
y_pred_classes = y_pred_proba > 0.5


# https://machinelearningmastery.com/how-to-calculate-precision-recall-f1-and-more-for-deep-learning-models/

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_gt_classes, y_pred_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_gt_classes, y_pred_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_gt_classes, y_pred_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_gt_classes, y_pred_classes)
print('F1 score: %f' % f1)

# ROC AUC
auc = roc_auc_score(y_gt_classes, y_pred_proba)
print('ROC AUC: %f' % auc)


precision, recall, _ = precision_recall_curve(y_gt_classes, y_pred_proba)
fpr, tpr, thresholds = roc_curve(y_gt_classes, y_pred_proba)

f, axes = plt.subplots(1, 2, sharex=True, sharey=True)
f.set_size_inches((8, 4))
axes[0].fill_between(recall, precision, step='post', alpha=0.2, color='b')
axes[0].set_title('Recall-Precision Curve')
axes[1].plot(fpr, tpr)
axes[1].plot([0, 1], [0, 1], linestyle='--')
axes[1].set_title('ROC curve')
plt.show()

for i, (image_file, gt, proba) in enumerate(zip(test_generator.filenames, y_gt_classes, y_pred_proba)):
    if abs(gt - proba) > 0.5:
        print('%d - %s : %f' % (i, image_file, proba))
        image_file_path = os.path.join(TEST_DATASET_DIR_PATH, image_file)
        image = np.array(Image.open(image_file_path))

        plt.figure(0)
        plt.clf()
        plt.suptitle('%s' % image_file)
        plt.title('predict=%.4f, label=%d' % (proba, gt))
        plt.imshow(image, cmap='gray')
        plt.show(block=False)
        plt.pause(1)
