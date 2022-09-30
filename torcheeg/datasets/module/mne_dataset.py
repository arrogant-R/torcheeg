from typing import Callable, Dict, List, Tuple, Union

import mne

from ..functional.mne import mne_constructor
from .base_dataset import BaseDataset


class MNEDataset(BaseDataset):
    r'''
    A generic EEG analysis dataset that allows creating datasets from :obj:`MNE.Epochs`, and caches the generated results in a unified input and output format (IO). It is generally used to support custom datasets or datasets not yet provided by TorchEEG.

    :obj:`MNEDataset` allows a list of :obj:`MNE.Epochs` and the corresponding description dictionary as input, and divides the signals in :obj:`MNE.Epochs` into several segments according to the configuration information provided by the user. These segments will be annotated by the description dictionary elements and the event type annotated in :obj:`MNE.Epochs`. Here is an example case shows the use of :obj:`MNEDataset`:

    .. code-block:: python
    
        # subject index and run index of MNE.Epochs
        metadata_list = [{
            'subject': 1,
            'run': 3
        }, {
            'subject': 1,
            'run': 7
        }, {
            'subject': 1,
            'run': 11
        }]

        epochs_list = []
        for metadata in metadata_list:
            physionet_path = mne.datasets.eegbci.load_data(metadata['subject'],
                                                        metadata['run'],
                                                        update_path=False)[0]

            raw = mne.io.read_raw_edf(physionet_path, preload=True, stim_channel='auto')
            events, _ = mne.events_from_annotations(raw)
            picks = mne.pick_types(raw.info,
                                meg=False,
                                eeg=True,
                                stim=False,
                                eog=False,
                                exclude='bads')
            # init Epochs with raw EEG signals and corresponding event annotations
            epochs_list.append(mne.Epochs(raw, events, picks=picks))

        # split into chunks of 160 data points (1s)
        dataset = MNEDataset(epochs_list=epochs_list,
                            metadata_list=metadata_list,
                            chunk_size=160,
                            overlap=0,
                            num_channel=60,
                            io_path=io_path,
                            offline_transform=transforms.Compose(
                                [transforms.BandDifferentialEntropy()]),
                            online_transform=transforms.ToTensor(),
                            label_transform=transforms.Compose([
                                transforms.Select('event')
                            ]),
                            num_worker=2)
        print(dataset[0])
        # EEG signal (torch.Tensor[60, 4]),
        # coresponding baseline signal (torch.Tensor[60, 4]),
        # label (int)

    In particular, TorchEEG utilizes the producer-consumer model to allow multi-process data preprocessing. If your data preprocessing is time consuming, consider increasing :obj:`num_worker` for higher speedup. If running under Windows, please use the proper idiom in the main module:

    .. code-block:: python
    
        if __name__ == '__main__':
            # subject index and run index of MNE.Epochs
            metadata_list = [{
                'subject': 1,
                'run': 3
            }, {
                'subject': 1,
                'run': 7
            }, {
                'subject': 1,
                'run': 11
            }]

            epochs_list = []
            for metadata in metadata_list:
                physionet_path = mne.datasets.eegbci.load_data(metadata['subject'],
                                                            metadata['run'],
                                                            update_path=False)[0]

                raw = mne.io.read_raw_edf(physionet_path, preload=True, stim_channel='auto')
                events, _ = mne.events_from_annotations(raw)
                picks = mne.pick_types(raw.info,
                                    meg=False,
                                    eeg=True,
                                    stim=False,
                                    eog=False,
                                    exclude='bads')
                # init Epochs with raw EEG signals and corresponding event annotations
                epochs_list.append(mne.Epochs(raw, events, picks=picks))

            # split into chunks of 160 data points (1s)
            dataset = MNEDataset(epochs_list=epochs_list,
                                metadata_list=metadata_list,
                                chunk_size=160,
                                overlap=0,
                                num_channel=60,
                                io_path=io_path,
                                offline_transform=transforms.Compose(
                                    [transforms.BandDifferentialEntropy()]),
                                online_transform=transforms.ToTensor(),
                                label_transform=transforms.Compose([
                                    transforms.Select('event')
                                ]),
                                num_worker=2)
            print(dataset[0])
            # EEG signal (torch.Tensor[60, 4]),
            # coresponding baseline signal (torch.Tensor[60, 4]),
            # label (int)

    Args:
        epochs_list (list): A list of :obj:`MNE.Epochs`. :obj:`MNEDataset` will divide the signals in :obj:`MNE.Epochs` into several segments according to the :obj:`chunk_size` and :obj:`overlap` information provided by the user. The divided segments will be transformed and cached in a unified input and output format (IO) for accessing.
        metadata_list (list): A list of dictionaries of the same length as :obj:`epochs_list`. Each of these dictionaries is annotated with meta-information about :obj:`MNE.Epochs`, such as subject index, experimental dates, etc. These annotated meta-information will be added to the element corresponding to :obj:`MNE.Epochs` for use as labels for the sample.
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. If set to -1, the EEG signal is not segmented, and the length of the chunk is the length of the event. (default: :obj:`-1`)
        overlap (int): The number of overlapping data points between different chunks when dividing EEG chunks. (default: :obj:`0`)
        num_channel (int): Number of channels used. If set to -1, all electrodes are used (default: :obj:`-1`)
        online_transform (Callable, optional): The transformation of the EEG signals and baseline EEG signals. The input is a :obj:`np.ndarray`, and the ouput is used as the first and second value of each element in the dataset. (default: :obj:`None`)
        offline_transform (Callable, optional): The usage is the same as :obj:`online_transform`, but executed before generating IO intermediate results. (default: :obj:`None`)
        label_transform (Callable, optional): The transformation of the label. The input is an information dictionary, and the ouput is used as the third value of each element in the dataset. (default: :obj:`None`)
        io_path (str): The path to generated unified data IO, cached as an intermediate result. (default: :obj:`./io/deap`)
        num_worker (str): How many subprocesses to use for data processing. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
        cache_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`64 * 1024 * 1024 * 1024`)
    
    '''
    def __init__(self,
                 epochs_list: List[mne.Epochs],
                 metadata_list: List[Dict],
                 chunk_size: int = -1,
                 overlap: int = 0,
                 num_channel: int = -1,
                 online_transform: Union[None, Callable] = None,
                 offline_transform: Union[None, Callable] = None,
                 label_transform: Union[None, Callable] = None,
                 io_path: str = './io/mne',
                 num_worker: int = 0,
                 verbose: bool = True,
                 cache_size: int = 64 * 1024 * 1024 * 1024):
        mne_constructor(epochs_list=epochs_list,
                        metadata_list=metadata_list,
                        chunk_size=chunk_size,
                        overlap=overlap,
                        num_channel=num_channel,
                        transform=offline_transform,
                        io_path=io_path,
                        num_worker=num_worker,
                        verbose=verbose,
                        cache_size=cache_size)
        super().__init__(io_path)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.num_channel = num_channel
        self.online_transform = online_transform
        self.offline_transform = offline_transform
        self.label_transform = label_transform
        self.num_worker = num_worker
        self.verbose = verbose
        self.cache_size = cache_size

    def __getitem__(self, index: int) -> Tuple:
        info = self.info.iloc[index].to_dict()

        eeg_index = str(info['clip_id'])
        eeg = self.eeg_io.read_eeg(eeg_index)

        signal = eeg
        label = info

        if self.online_transform:
            signal = self.online_transform(eeg=eeg)['eeg']

        if self.label_transform:
            label = self.label_transform(y=info)['y']

        return signal, label

    @property
    def repr_body(self) -> Dict:
        return dict(
            super().repr_body, **{
                'chunk_size': self.chunk_size,
                'overlap': self.overlap,
                'num_channel': self.num_channel,
                'online_transform': self.online_transform,
                'offline_transform': self.offline_transform,
                'label_transform': self.label_transform,
                'num_worker': self.num_worker,
                'verbose': self.verbose,
                'cache_size': self.cache_size
            })