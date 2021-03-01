from __future__ import print_function, division

# Stop matplotlib from drawing to X.
# Must be before importing matplotlib.pyplot or pylab!
import matplotlib
matplotlib.use('Agg')

from lasagne.layers import (InputLayer, DenseLayer, ReshapeLayer,
                            DimshuffleLayer, Conv1DLayer)
from lasagne.nonlinearities import rectify

from neuralnilm.data.loadactivations import load_nilmtk_activations
from neuralnilm.data.syntheticaggregatesource import SyntheticAggregateSource
from neuralnilm.data.realaggregatesource import RealAggregateSource
from neuralnilm.data.stridesource import StrideSource
from neuralnilm.data.datapipeline import DataPipeline
from neuralnilm.data.processing import DivideBy, IndependentlyCenter
from neuralnilm.net import Net, build_net
from neuralnilm.trainer import Trainer
from neuralnilm.metrics import Metrics
from neuralnilm.consts import DATA_FOLD_NAMES


NILMTK_FILENAME = '/data/dk3810/ukdale.h5'
SAMPLE_PERIOD = 6
STRIDE = None
APPLIANCES = [
    'kettle', 'microwave', 'washing machine', 'dish washer', 'fridge']
WINDOWS = {
    'train': {
        1: ("2013-04-12", "2015-07-01"),
        2: ("2013-05-22", "2013-10-03 06:16:00"),
        3: ("2013-02-27", "2013-04-01 06:15:05"),
        4: ("2013-03-09", "2013-09-24 06:15:14")
    },
    'unseen_activations_of_seen_appliances': {
        1: ("2015-07-02", None),
        2: ("2013-10-03 06:16:00", None),
        3: ("2013-04-01 06:15:05", None),
        4: ("2013-09-24 06:15:14", None)
    },
    'unseen_appliances': {
        5: ("2014-06-29", None)
    }
}


def run(root_experiment_name):
    activations = load_nilmtk_activations(
        appliances=APPLIANCES,
        filename=NILMTK_FILENAME,
        sample_period=SAMPLE_PERIOD,
        windows=WINDOWS
    )

    for get_net in [ae]:
        for target_appliance in APPLIANCES:
            pipeline = get_pipeline(target_appliance, activations)

            # Build net
            batch = pipeline.get_batch()
            net = get_net(batch)

            # Trainer
            trainer = Trainer(
                net=net,
                data_pipeline=pipeline,
                experiment_id=[
                    root_experiment_name, get_net.__name__, target_appliance],
                metrics=Metrics(state_boundaries=[2.5]),
                learning_rates={
                    0: 1e-1,
                    50000: 1e-2,
                    200000: 1e-3
                },
                repeat_callbacks=[
                    (25000, Trainer.validate),
                    (25000, Trainer.save_params),
                    (25000, Trainer.plot_estimates)
                ]
            )

            report = trainer.submit_report()
            print(report)

            # Run!
            trainer.fit(300000)


# ----------------------  NETWORKS  -------------------------
def ae(batch):
    NUM_FILTERS = 8
    input_shape = batch.input.shape
    seq_length = input_shape[1]
    output_layer = build_net(
        input_shape=input_shape,
        layers=[
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)  # (batch, features, time)
            },
            {
                'type': Conv1DLayer,  # convolve over the time axis
                'num_filters': NUM_FILTERS,
                'filter_size': 4,
                'stride': 1,
                'nonlinearity': None,
                'pad': 'valid'
            },
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)  # back to (batch, time, features)
            },
            {
                'type': DenseLayer,
                'num_units': (seq_length - 3) * NUM_FILTERS,
                'nonlinearity': rectify
            },
            {
                'type': DenseLayer,
                'num_units': 128,
                'nonlinearity': rectify
            },
            {
                'type': DenseLayer,
                'num_units': (seq_length - 3) * NUM_FILTERS,
                'nonlinearity': rectify
            },
            {
                'type': ReshapeLayer,
                'shape': (-1, (seq_length - 3), NUM_FILTERS)
            },
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)  # (batch, features, time)
            },
            {   # DeConv
                'type': Conv1DLayer,
                'num_filters': 1,
                'filter_size': 4,
                'stride': 1,
                'nonlinearity': None,
                'pad': 'full'
            },
            {
                'type': DimshuffleLayer,
                'pattern': (0, 2, 1)  # back to (batch, time, features)
            }
        ]
    )

    net = Net(
        output_layer,
        tags=['AE', 'Conv1D'],
        description="Identical AE to e567.  Trained on 5 appliances.",
        predecessor_experiment="e567"
    )
    return net


# ------------------------ DATA ----------------------

def get_pipeline(target_appliance, activations):

    num_seq_per_batch = 64
    if target_appliance == 'kettle':
        seq_length = 128
        train_buildings = [1, 2, 4]
        unseen_buildings = [5]
    elif target_appliance == 'microwave':
        seq_length = 288
        train_buildings = [1, 2]
        unseen_buildings = [5]
    elif target_appliance == 'washing machine':
        seq_length = 1024
        train_buildings = [1, 5]
        unseen_buildings = [2]
    elif target_appliance == 'fridge':
        seq_length = 512
        train_buildings = [1, 2, 4]
        unseen_buildings = [5]
    elif target_appliance == 'dish washer':
        seq_length = 1024 + 512
        train_buildings = [1, 2]
        unseen_buildings = [5]

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

    stride_source = StrideSource(
        target_appliance=target_appliance,
        seq_length=seq_length,
        filename=NILMTK_FILENAME,
        windows=filtered_windows,
        sample_period=SAMPLE_PERIOD,
        stride=STRIDE
    )

    sample = real_agg_source.get_batch(num_seq_per_batch=1024).next()
    sample = sample.before_processing
    input_std = sample.input.flatten().std()
    target_std = sample.target.flatten().std()
    pipeline = DataPipeline(
        [synthetic_agg_source, real_agg_source, stride_source],
        num_seq_per_batch=num_seq_per_batch,
        input_processing=[DivideBy(input_std), IndependentlyCenter()],
        target_processing=[DivideBy(target_std)]
    )

    return pipeline


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
