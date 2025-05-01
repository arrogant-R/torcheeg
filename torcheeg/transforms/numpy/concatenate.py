from typing import Callable, Sequence, Dict, Union

import numpy as np

from ..base_transform import EEGTransform


class Concatenate(EEGTransform):
    r'''
    Merge the calculation results of multiple transforms, which are used when feature fusion is required.

    .. code-block:: python

        from torcheeg import transforms

        t = transforms.Concatenate([
            transforms.BandDifferentialEntropy(),
            transforms.BandMeanAbsoluteDeviation()
        ])
        t(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (32, 8)

    Args:
        transforms (list, tuple): a sequence of transforms.
        axis (int): The axis along which the arrays will be joined. If axis is None, arrays are flattened before use (default: :obj:`-1`).
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)
    
    .. automethod:: __call__
    '''
    def __init__(self,
                 transforms: Sequence[Callable],
                 axis: int = -1,
                 apply_to_baseline: bool = False):
        super(Concatenate, self).__init__(apply_to_baseline=apply_to_baseline)
        self.transforms = transforms
        self.axis = axis

    def __call__(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        r'''
        Args:
            eeg (np.ndarray): The input EEG signals in shape of [number of electrodes, number of data points].
            baseline (np.ndarray, optional) : The corresponding baseline signal, if apply_to_baseline is set to True and baseline is passed, the baseline signal will be transformed with the same way as the experimental signal.
            
        Returns:
            np.ndarray: The combined results of multiple transforms.
        '''
        if args:
            raise KeyError("Please pass data as named parameters.")

        target_kwargs = {}
        non_target_kwargs = {}

        for t in self.transforms:
            new_kwargs_t = t(**kwargs)
            for new_kwargs_key, new_kwargs_value in new_kwargs_t.items():
                if not new_kwargs_key in self.targets:
                    non_target_kwargs[new_kwargs_key] = new_kwargs_value
                    continue
                assert isinstance(
                    new_kwargs_value, np.ndarray
                ), f'Concate only supports concatenating numpy.ndarray type data, you are trying to concatenate {type(new_kwargs_value)} type data.'
                if not new_kwargs_key in target_kwargs:
                    target_kwargs[new_kwargs_key] = []
                target_kwargs[new_kwargs_key].append(new_kwargs_value)

        for target_kwargs_key, target_kwargs_value in target_kwargs.items():
            target_kwargs[target_kwargs_key] = np.concatenate(
                target_kwargs_value, axis=self.axis)

        target_kwargs.update(non_target_kwargs)
        return target_kwargs

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '('
        for i, t in enumerate(self.transforms):
            if i:
                format_string += ','
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


class MapChunk(EEGTransform):
    r'''
    Divide the input EEG signal into multiple chunks according to chunk_size and overlap, and then apply a transofrm to each chunk, and combine the calculation results of a transofrm on all chunks. It is used when feature fusion is required.

    .. code-block:: python

        from torcheeg import transforms

        t = transforms.MapChunk(
            transforms.BandDifferentialEntropy(),
            chunk_size=250,
            overlap=0
        )
        t(eeg=np.random.randn(64, 1000))['eeg'].shape
        >>> (64, 16)

    TorchEEG allows feature fusion at multiple scales:

    .. code-block:: python

        from torcheeg import transforms

        t = transforms.Concatenate([
            transforms.MapChunk(
                transforms.BandDifferentialEntropy()
                chunk_size=250,
                overlap=0),  # 4 chunk * 4-dim feature
            transforms.MapChunk(
                transforms.BandDifferentialEntropy()
                chunk_size=500,
                overlap=0),  # 2 chunk * 4-dim feature
            transforms.BandDifferentialEntropy()  # 1 chunk * 4-dim feature
        ])
        t(eeg=np.random.randn(64, 1000))['eeg'].shape
        >>> (64, 28) # 4 * 4 + 2 * 4 + 1 * 4

    Args:
        transform (EEGTransform): a transform
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)
    
    .. automethod:: __call__
    '''
    def __init__(self,
                 transform: EEGTransform,
                 chunk_size: int = 250,
                 overlap: int = 0,
                 stack:bool = False,
                 chunk_dim = -1,
                 apply_to_baseline: bool = False):
        super(MapChunk,
              self).__init__(apply_to_baseline=apply_to_baseline)
        self.transform = transform
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.stack = stack
        self.chunk_dim = chunk_dim

    def __call__(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        r'''
        Args:
            eeg (np.ndarray): The input EEG signals in shape of [number of electrodes, number of data points].
            baseline (np.ndarray, optional) : The corresponding baseline signal, if apply_to_baseline is set to True and baseline is passed, the baseline signal will be transformed with the same way as the experimental signal.
            
        Returns:
            np.ndarray: The combined results of a transform from multiple chunks.
        '''
        
        target_kwargs = {}
        non_target_kwargs = {}

        # check_len_key = None
        # check_len_value = None

        for kwargs_key_org, kwargs_value in kwargs.items():

            kwargs_key = self._additional_targets.get(kwargs_key_org,kwargs_key_org)

            if not kwargs_key in self.targets:
                non_target_kwargs[kwargs_key] = kwargs_value
                continue

            # if not check_len_key:
            #     check_len_key = kwargs_key
            #     check_len_value = len(kwargs_value)
            # else:
            #     assert len(
            #         kwargs_value
            #     ) == check_len_value, f'The lengths of {check_len_key} ({check_len_value}) and {kwargs_key} ({len(kwargs_value)}) in the input signal are not the same.'

            start_at = 0
            end_at = start_at + self.chunk_size
            step = self.chunk_size - self.overlap

            chunk_kwargs_value = []
            
            chunk_dim = kwargs_value.ndim + self.chunk_dim if self.chunk_dim < 0 else self.chunk_dim
            while end_at <= kwargs_value.shape[self.chunk_dim]:
                index =  tuple(slice(None) if i !=chunk_dim else range(start_at,end_at) for i in range(kwargs_value.ndim))
                chunk_kwargs_value.append(kwargs_value[index])
                start_at = start_at + step
                end_at = start_at + self.chunk_size
            target_kwargs[kwargs_key_org] = chunk_kwargs_value
            

        new_target_kwargs = {}
        for target_key,target_value in target_kwargs.items():
            num_chunk = len(target_value)
            transform_target_key = self._additional_targets.get(target_key,target_key)
            for chunk_index in range(num_chunk):
                cur_target_kwargs = {
                  transform_target_key: target_value[chunk_index]
                }
                
                cur_target_kwargs.update(non_target_kwargs)

                new_kwargs_t = self.transform(**cur_target_kwargs)

                for new_kwargs_key, new_kwargs_value in new_kwargs_t.items():
                    if not new_kwargs_key == transform_target_key:
                        continue
                    assert isinstance(
                        new_kwargs_value, np.ndarray
                    ), f'Concate only supports concatenating numpy.ndarray type data, you are trying to concatenate {type(new_kwargs_value)} type data.'
                    if not target_key in new_target_kwargs:
                        new_target_kwargs[target_key] = []
                        
                    new_target_kwargs[target_key].append(new_kwargs_value)

        for new_target_kwargs_key, new_target_kwargs_value in new_target_kwargs.items(
        ):
            if self.stack:
               
                new_target_kwargs[new_target_kwargs_key] = np.stack(
                new_target_kwargs_value)
            else:
                new_target_kwargs[new_target_kwargs_key] = np.concatenate(
                    new_target_kwargs_value, axis=self.chunk_dim)
        new_target_kwargs.update(non_target_kwargs)
        return new_target_kwargs

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '(['
        for i, t in enumerate(self.transforms):
            if i:
                format_string += ','
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n],'
        format_string += f'\nchunk_size={self.chunk_size}, '
        format_string += f'\noverlap={self.overlap})'
        return format_string
