from __future__ import print_function
from matplotlib.pylab import plt
from matplotlib import rcParams
rcParams['figure.figsize'] = (13, 6)
plt.style.use('ggplot')
import numpy as np
import nilmtk
from nilmtk.utils import print_dict
from nilmtk import DataSet
from neuralnilm.data.loadactivations import load_nilmtk_activations
from neuralnilm.data.syntheticaggregatesource import SyntheticAggregateSource
from neuralnilm.data.realaggregatesource import RealAggregateSource
from neuralnilm.data.stridesource import StrideSource
from neuralnilm.data.datapipeline import DataPipeline
from neuralnilm.data.processing import DivideBy, IndependentlyCenter

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D
from keras.optimizers import SGD
import time


# create dictionary with train, unseen_house, unseen_appliance
def select_windows(train_buildings, unseen_buildings):
    windows = {fold: {} for fold in DATA_FOLD_NAMES}

    def copy_window(fold, i):
        windows[fold][i] = WINDOWS[fold][i]

    for i in train_buildings:
        copy_window('train', i)
        copy_window('unseen_activations_of_seen_appliances', i)
    for i in unseen_buildings:
        copy_window('unseen_appliances', i)
    return windows


def filter_activations(windows, activations):
    new_activations = {
        fold: {appliance: {} for appliance in APPLIANCES}
        for fold in DATA_FOLD_NAMES}
    for fold, appliances in activations.iteritems():
        for appliance, buildings in appliances.iteritems():
            required_building_ids = windows[fold].keys()
            required_building_names = [
                'UK-DALE_building_{}'.format(i) for i in required_building_ids]
            for building_name in required_building_names:
                try:
                    new_activations[fold][appliance][building_name] = (
                        activations[fold][appliance][building_name])
                except KeyError:
                    pass
    return activations    


NILMTK_FILENAME = './redd_data/redd.h5'
SAMPLE_PERIOD = 6
STRIDE = None
APPLIANCES = ['fridge']
WINDOWS = {
    'train': {
        1: ("2011-04-19", "2011-05-21"),
        2: ("2011-04-19", "2013-05-01"),
        3: ("2011-04-19", "2013-05-26"),
        6: ("2011-05-22", "2011-06-14"),
    },
    'unseen_activations_of_seen_appliances': {
        1: ("2011-04-19", None),
        2: ("2011-04-19", None),
        3: ("2011-04-19", None),
        6: ("2011-05-22", None),
    },
    'unseen_appliances': {
        5: ("2011-04-19", None)
    }
}

# get the dictionary of activations for each appliance
activations = load_nilmtk_activations(
    appliances=APPLIANCES,
    filename=NILMTK_FILENAME,
    sample_period=SAMPLE_PERIOD,
    windows=WINDOWS
)

# get pipeline for the fridge example
num_test_seq = 101
num_seq_per_batch = 32
target_appliance = 'fridge'
seq_length = 512
train_buildings = [1, 2, 3, 6]
unseen_buildings = [5]
DATA_FOLD_NAMES = (
    'train', 'unseen_appliances', 'unseen_activations_of_seen_appliances')

filtered_windows = select_windows(train_buildings, unseen_buildings)
filtered_activations = filter_activations(filtered_windows, activations)

synthetic_agg_source = SyntheticAggregateSource(
    activations=filtered_activations,
    target_appliance=target_appliance,
    seq_length=seq_length,
    sample_period=SAMPLE_PERIOD
)

real_agg_source = RealAggregateSource(
    activations=filtered_activations,
    target_appliance=target_appliance,
    seq_length=seq_length,
    filename=NILMTK_FILENAME,
    windows=filtered_windows,
    sample_period=SAMPLE_PERIOD
)


# ------------
# needed to rescale the input aggregated data
# rescaling is done using the a first batch of num_seq_per_batch sequences
sample = real_agg_source.get_batch(num_seq_per_batch=1024).next()
sample = sample.before_processing
input_std = sample.input.flatten().std()
target_std = sample.target.flatten().std()
# ------------



pipeline = DataPipeline(
    [synthetic_agg_source, real_agg_source],
    num_seq_per_batch=num_seq_per_batch,
    input_processing=[DivideBy(input_std), IndependentlyCenter()],
    target_processing=[DivideBy(target_std)]
)


from keras.layers import Input, Dense, Flatten
from keras.models import Model

starting_time = time.time()

# define the network architecture = Conv Net
input_seq = Input(shape = (1, seq_length))
conv1_layer =  Convolution1D(nb_filter = 16, filter_length = 3, border_mode='same',
                      init = 'normal', activation =  'relu')
conv1 = conv1_layer(input_seq)
conv2 = Convolution1D(nb_filter = 16, filter_length = 3, border_mode='same',
                      init = 'normal', activation =  'relu')(conv1)
flat = Flatten()(conv2)
# dense1 = Dense(4080, activation = 'relu')(flat)
# dense2 = Dense(3072, activation = 'relu')(flat)
dense3 = Dense(2048, activation = 'relu')(flat)
dense4 = Dense(512, activation = 'relu')(dense3)
predictions = Dense(3, activation = 'relu')(dense4)
# create the model
model = Model(input=input_seq, output=predictions)
# compile the model
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error',
              optimizer='Adam')
compiling_time = time.time() - starting_time
print('compiling time = ', compiling_time)
history = model.fit_generator(pipeline.train_generator(fold = 'train'), \
                    samples_per_epoch = num_seq_per_batch*500, \
                    nb_epoch = 500, verbose = 1)
print('run time = ', time.time() - starting_time)





# -----
# saving the model
from keras.models import model_from_json
json_string = model.to_json()
open('./tmpdata/convnet_architecture.json', 'w').write(json_string)
print('saving the weights ... ')
model.save_weights('./tmpdata/my_model_weights.h5', overwrite = True)
print('done saving the weights')



# prediction
# number of sample
from neuralnilm.metrics import Metrics
score = Metrics(state_boundaries = [2.5])
Y_test = np.empty((num_seq_per_batch*num_test_seq,3))
Y_pred = np.empty((num_seq_per_batch*num_test_seq,3))
for i in range(num_test_seq):
    (x_test, y_test) = pipeline.train_generator(fold = 'unseen_appliances', source_id = 1).next()
    Y_test[num_seq_per_batch*i: num_seq_per_batch*(i+1)] = y_test
    Y_pred[num_seq_per_batch*i: num_seq_per_batch*(i+1)] = model.predict_on_batch(x_test)
results = score.compute_metrics(Y_pred, Y_test)
print
print(results)