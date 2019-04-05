import pickle
import numpy as np
import chainer.functions as F
import chainer.iterators as I
import chainer.optimizers as O
import chainer.serializers as S
import chainer.training  as T
import chainer.training.extensions as E
import datetime
from ONT import *
from ONT_Classifier import *



''' Loading '''
# Parameter
size_batch = 256
rate_train_data = 0.8

#
with open('../pickle/data.pickle', 'rb') as f:
    data = pickle.load(f)
#
n_data = len(data)
n_train = round(n_data * rate_train_data)
n_valid = n_data - n_train
#
train_data = data[:n_train]
valid_data = data[n_train:]
# Iterator
train_iter = I.SerialIterator(train_data, size_batch, shuffle=True)
valid_iter = I.SerialIterator(valid_data, size_batch, repeat=False)
# MemoryError対策
del data
del train_data
del valid_data
#
print('Training   :', n_train)
print('Validation :', n_valid)
print('Total      :', n_data)


''' Model '''
# Parameter
n_rnn = 1
size_hidden = 300
rate_dropout = 0.1


# Network & Model
net = ONT_LSTM(n_rnn, size_hidden, rate_dropout)
model = ONT_Classifier(net)
# Optimizer
optimizer = O.Adam()
optimizer.setup(model)


''' Trainer '''
# property
stop_trigger = (150, 'epoch')
out_dir = '../result/' + datetime.datetime.now().isoformat()
#   log
log_trigger = (1, 'epoch')
print_entries = ['epoch','main/loss', 'validation/main/loss',
                         'main/accu', 'validation/main/accu']
#   plot
# plot_trigger = (1, 'epoch')
# loss_fname = 'loss.png'
# loss_entries = ['main/loss', 'validation/main/loss']
# accu_fname = 'accu.png'
# accu_entries = ['main/accu', 'validation/main/accu']
#   save
save_trigger = (1, 'epoch')
train_fname = 'train_epoch-{.updater.epoch}'
model_fname = 'model_epoch-{.updater.epoch}'

# Trainer
updater = T.StandardUpdater(train_iter, optimizer, converter=model.converter)
trainer = T.Trainer(updater, stop_trigger, out_dir)

trainer.extend(E.Evaluator(valid_iter, model, converter=model.converter))
trainer.extend(E.LogReport(trigger=log_trigger))
# trainer.extend(E.PrintReport(print_entries))
# trainer.extend(E.PlotReport(loss_entries, 'epoch', file_name=loss_fname), trigger=plot_trigger)
# trainer.extend(E.PlotReport(accu_entries, 'epoch', file_name=accu_fname), trigger=plot_trigger)
trainer.extend(E.snapshot(              filename=train_fname), trigger=save_trigger)
trainer.extend(E.snapshot_object(model, filename=model_fname), trigger=save_trigger)
trainer.extend(E.ProgressBar())
trainer.run()
