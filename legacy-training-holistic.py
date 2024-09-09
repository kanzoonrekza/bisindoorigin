# %% [markdown]
# # Code for training the holistic model

# %% [markdown]
# ## Import Libraries
#

# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, log_loss
import io
import sys
import tensorflow as tf
import numpy as np
import os
from datetime import datetime

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
from keras.optimizers import Adam


from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, log_loss, classification_report

# Variables
n = 2  # Data duplication
handsOnly = True  # Whether to use only hands or not
learning_rate = 0.0001
epoch = 100

FOLDER_NAME = 'dataset'
ALL_CLASSES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
               'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# %%
# Create label map, representing each class as a number
label_map = {}
for (root, folders, files) in os.walk(FOLDER_NAME):
    for foldername in folders:
        if foldername in ALL_CLASSES:
            label_map[foldername] = ALL_CLASSES.index(foldername)

print(label_map)

# %%
# Get all datset data with its label and put it in a list
sequence, label = [], []
target_length = 14
for (root, folders, files) in os.walk(FOLDER_NAME):
    total_file = 0
    for filename in files:
        file_path = os.path.join(os.path.relpath(
            root, FOLDER_NAME), filename)
        if (filename.endswith('.npy') and os.path.split(file_path)[0] in ALL_CLASSES):
            res = np.load(f'{FOLDER_NAME}/{file_path}')
            for _ in range(target_length-res.shape[0]):
                res = np.vstack((res, res[-1, :]))
            if (handsOnly):
                res = res[:, -126:]
            sequence.append(np.array(res))
            label.append(label_map[os.path.basename(root[-1])])
            total_file += 1
    print(f"Total files: {total_file} --- {root}")

print(np.array(sequence).shape)
print(np.array(label).shape)

# %%
sequence = np.concatenate([sequence] * n, axis=0)
label = np.concatenate([label] * n, axis=0)


print(np.array(sequence).shape)
print(np.array(label).shape)

# %%
tf.config.list_physical_devices('GPU')


# %% [markdown]
# ## Training Data

# %%
X_train, X_test, y_train, y_test = train_test_split(np.array(sequence), tf.keras.utils.to_categorical(
    np.array(label).astype(int), num_classes=np.array(ALL_CLASSES).shape[0], dtype='float32'), test_size=0.2)

print(X_train.shape, X_test.shape)


# %%
training_phase = str(np.array(sequence).shape[2]) + "-tanh-lr-" + str(learning_rate).replace(
    "0.", "") + "-dupli-" + str(n) + "-" + str(epoch) + "-epoch-" + datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join('Logs', training_phase)
tb_callback = TensorBoard(log_dir=log_dir)

print(log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True,
               activation='tanh', input_shape=(14, np.array(sequence).shape[2])))
model.add(LSTM(64, return_sequences=True, activation='tanh'))
model.add(LSTM(64, return_sequences=False, activation='tanh'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(np.array(ALL_CLASSES).shape[0], activation='softmax'))

optimizer = Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer, loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

# %%
model.fit(X_train, y_train, epochs=epoch, callbacks=[
          tb_callback])

# %%
model.summary()


# %%
# Save model
model.save(f'{log_dir}/action.h5')

# %%
res = model.predict(X_test)
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calculate 
accuracy = accuracy_score(y_true, y_pred_classes)
precision = precision_score(y_true, y_pred_classes, average='weighted')
recall = recall_score(y_true, y_pred_classes, average='weighted')
f1 = f1_score(y_true, y_pred_classes, average='weighted')
classes = np.unique(y_true)
loss = log_loss(y_true, y_pred, labels=classes)

# Redirect stdout to a string buffer
old_stdout = sys.stdout
sys.stdout = buffer = io.StringIO()

print(f"Training Phase: {training_phase}\n\n")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Loss: {loss}")

# Detailed classification report
report = classification_report(y_true, y_pred_classes)

# Restore stdout
sys.stdout = old_stdout
output = buffer.getvalue()

phase_dir = f'Logs/{training_phase}'
log_filename = f'{phase_dir}/summary.txt'
with open(log_filename, 'w') as f:
    f.write(output)
    f.write("\n")
    f.write(report)

# Optionally, print the output to console as well
print(output)


# %%
actions = np.array(ALL_CLASSES)
testing = 13
print(actions[np.argmax(res[testing])], actions[np.argmax(y_test[testing])])
