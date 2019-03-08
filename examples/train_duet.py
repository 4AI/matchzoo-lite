# -*- coding: utf-8 -*-

import os
import sys
from pprint import pprint
import matchzoo as mz
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

if len(sys.argv) != 2:
    print("usage: python3 train_duet.py [corpus_dir]")
    exit()

corpus_dir = sys.argv[1]
models_dir = 'models/duet'
max_sentence_len = 100
hparams = {
    'n_epochs': 10,
    'batch_size': 32,
}

train_pack, _ = mz.dataloader.load_data('train', task='classification', corpus_dir=corpus_dir)
valid_pack, _ = mz.dataloader.load_data('test', task='classification', corpus_dir=corpus_dir)

print("start to prepro...")
preprocessor = mz.preprocessors.BasicPreprocessor(fixed_length_left=max_sentence_len,
                                                  fixed_length_right=max_sentence_len,
                                                  remove_stop_words=False)
train_pack_processed = preprocessor.fit_transform(train_pack, verbose=0)
preprocessor.save(models_dir)
valid_pack_processed = preprocessor.transform(valid_pack, verbose=0)
print("Done with preprocessing")

# prepare task
classification_task = mz.tasks.Classification(num_classes=2)
# prepare model
model = mz.models.DUET()
model.params['input_shapes'] = preprocessor.context['input_shapes']
model.params['task'] = classification_task
model.params['embedding_input_dim'] = preprocessor.context['vocab_size']
model.params['embedding_output_dim'] = 100
model.params['embedding_trainable'] = True
model.params['lm_filters'] = 128
model.params['lm_hidden_sizes'] = [256, 128]
model.params['dm_filters'] = 64
model.params['dm_kernel_size'] = 3
model.params['dm_d_mpool'] = 4
model.params['dm_q_hidden_size'] = 64
model.params['dm_hidden_sizes'] = [128, 64]
model.params['dropout_rate'] = 0.5
model.params['optimizer'] = 'adam'
model.guess_and_fill_missing_params()
model.build()
model.compile()

print("hyper-parameters: ")
hparams = dict(model.params, **hparams)
pprint(hparams)

print("start to train...")
# train & save model
train_generator = mz.DataGenerator(train_pack_processed, batch_size=hparams['batch_size'], shuffle=True)
model.fit_generator(train_generator, epochs=hparams['n_epochs'], use_multiprocessing=False, workers=1, verbose=0)
model.save(models_dir)

# evaluate on test dataset
valid_x, valid_y = valid_pack_processed.unpack()
preds = model.predict(valid_x)
y_trues, y_preds = [], []
for pred, y in zip(preds, valid_y):
    y_true = y[1]
    y_trues.append(int(y[1]))
    y_pred = 0 if pred[1] < pred[0] else 1
    y_preds.append(y_pred)

print("precision: ", precision_score(y_trues, y_preds))
print("recall: ", recall_score(y_trues, y_preds))
print("f1: ", f1_score(y_trues, y_preds))
print(classification_report(y_trues, y_preds))
