from __future__ import print_function, division
from copy import copy
import numpy as np

from neuralnilm.utils import none_to_list


class DataPipeline(object):
    def __init__(self, sources, num_seq_per_batch,
                 input_processing=None,
                 target_processing=None,
                 source_probabilities=None,
                 rng_seed=None):
        self.sources = sources
        self.num_seq_per_batch = num_seq_per_batch
        self.input_processing = none_to_list(input_processing)
        self.target_processing = none_to_list(target_processing)
        num_sources = len(self.sources)
        if source_probabilities is None:
            self.source_probabilities = [1 / num_sources] * num_sources
        else:
            self.source_probabilities = source_probabilities
        self.rng_seed = rng_seed
        self.rng = np.random.RandomState(self.rng_seed)
        self._source_iterators = [None] * num_sources

    def get_batch(self, fold='train', enable_all_appliances=False,
                  source_id=None, reset_iterator=False,
                  validation=False):
        """
        Returns
        -------
        A Batch object or None if source iterator has hit a StopIteration.
        """
        if source_id is None:
            n = len(self.sources)
            source_id = self.rng.choice(n, p=self.source_probabilities)
        if reset_iterator or self._source_iterators[source_id] is None:
            self._source_iterators[source_id] = (
                self.sources[source_id].get_batch(
                    num_seq_per_batch=self.num_seq_per_batch,
                    fold=fold,
                    enable_all_appliances=enable_all_appliances,
                    validation=validation))
        try:
            batch = self._source_iterators[source_id].__next__()
        except StopIteration:
            self._source_iterators[source_id] = None
            return None
        else:
            batch.after_processing.input, i_metadata = self.apply_processing(
                batch.before_processing.input, 'input')
            batch.after_processing.target, t_metadata = self.apply_processing(
                batch.before_processing.target, 'target')
            batch.metadata.update({
                'source_id': source_id,
                'processing': {
                    'input': i_metadata,
                    'target': t_metadata
                }
            })
            return batch

    def apply_processing(self, data, net_input_or_target):
        """Applies `<input, target>_processing` to `data`.

        Parameters
        ----------
        data : np.ndarray
            shape = (num_seq_per_batch, seq_length, num_features)
        net_input_or_target : {'target', 'input}

        Returns
        -------
        processed_data, metadata
        processed_data : np.ndarray
            shape = (num_seq_per_batch, seq_length, num_features)
        metadata : dict
        """
        processing_steps = self._get_processing_steps(net_input_or_target)
        metadata = {}
        for step in processing_steps:
            data = step(data)
            if hasattr(step, 'metadata'):
                metadata.update(step.metadata)
        return data, metadata

    def apply_inverse_processing(self, data, net_input_or_target):
        """Applies the inverse of `<input, target>_processing` to `data`.

        Parameters
        ----------
        data : np.ndarray
            shape = (num_seq_per_batch, seq_length, num_features)
        net_input_or_target : {'target', 'input}

        Returns
        -------
        processed_data : np.ndarray
            shape = (num_seq_per_batch, seq_length, num_features)
        """
        processing_steps = self._get_processing_steps(net_input_or_target)
        reversed_processing_steps = processing_steps[::-1]
        for step in reversed_processing_steps:
            try:
                data = step.inverse(data)
            except AttributeError:
                pass
        return data

    def _get_processing_steps(self, net_input_or_target):
        assert net_input_or_target in ['input', 'target']
        attribute = net_input_or_target + '_processing'
        processing_steps = getattr(self, attribute)
        assert isinstance(processing_steps, list)
        return processing_steps

    def report(self):
        report = copy(self.__dict__)
        for attr in ['sources', 'rng', '_source_iterators']:
            report.pop(attr)
        report['sources'] = {
            i: source.report() for i, source in enumerate(self.sources)}
        report['input_processing'] = [
            processor.report() for processor in self.input_processing]
        report['target_processing'] = [
            processor.report() for processor in self.target_processing]
        return {'pipeline': report}

    def _get_output_neurons(self, new_batch):
        batch_size = new_batch.target.shape[0]
        neural_net_output = np.empty((batch_size, 3))
        
        for b in range(batch_size):
            seq =  new_batch.target[b]

            # case 1 and 2: if the signal start at 0
            if seq[0] > 0:
                start = 0
                stop_array = np.where(seq > 0)[0]
                # case 2: signal stops after 1
                # set stop to the last element
                if len(stop_array) == 0:
                    stop = seq[-1]
                # case 1: signal stops before 1
                else:
                    stop = stop_array[-1]  
                # calculate avg power
                avg_power =  np.mean(seq[start:stop + 1])

            # case 3: signal starts after 0 and before 1
            else:
                start_array = np.where(seq > 0)[0]
                if len(start_array) == 0:
                    # case 5: there is no signal in the window
                    start = 0
                    stop = 0
                    avg_power = 0
                else:
                    start = start_array[0]
                    # find stop
                    stop_array = np.where(seq > 0)[0]
                    # case 4: signal stops after 1
                    # set to the last element
                    if len(stop_array) == 0:
                        stop = seq[-1]
                    else:
                        stop = stop_array[-1]        
                    avg_power =  np.mean(seq[start:stop + 1])
                    
            start = start / float(new_batch.target.shape[1] - 1)
            stop = stop  / float(new_batch.target.shape[1] - 1)
            if stop < start:
                raise ValueError("start must be before stop in sequence {}".format(b))

            neural_net_output[b, :] = np.array([start, stop, avg_power])

        return neural_net_output        
    
    def train_generator(self, fold='train', enable_all_appliances=False,
                  source_id=None, reset_iterator=False,
                  validation=False ):
        while 1:
            batch_iter = self.get_batch(fold, enable_all_appliances, source_id, reset_iterator,validation)
            X_train = batch_iter.input
            input_dim = X_train.shape[1]
            Y_train = self._get_output_neurons(batch_iter)
            yield (np.reshape(X_train, [self.num_seq_per_batch, input_dim, 1]), Y_train.astype(np.float32))








