import torch
from datasets import concatenate_datasets, DatasetDict
from datasets import Dataset as Dataset_hf
import numpy as np
from torch.utils.data import Dataset
import gc
from collections import defaultdict
import pandas as pd
import inspect

def extract_keys(function, kwargs, remove_from_orignal=True):

    """ 
    Function which extract the keys of "function" from "kwargs" by inspecting the signature of "function".
    
    """
    
    signature = inspect.signature(function)
    function_keys = signature.parameters.keys()
    
    if remove_from_orignal:
        kwargs_fn = {k:kwargs.pop(k) for k in function_keys if k in kwargs}
    else:
        kwargs_fn = {k:kwargs[k] for k in function_keys if k in kwargs}
        
    return kwargs_fn



def match_with_batchsize(lim, batchsize):
    """ 
    Function used by modify_datasets below to match return the integer closest to lim 
    which is multiple of batchsize, i.e., lim%batchsize=0.
    
    """

    if lim%batchsize==0:
        return lim        
    else:
        return lim - lim%batchsize
    


def modify_datasets(dataset, labeled_fr=0.5, model_type='TriTrain',labeled1_frac=0.33, train_key="train",
                    label_column='label', unlabeled_labels=-1, batchsize=16):
    
    """ 
    Function to modify pyarrow based datasets (huggingface dataset) for testing 
    at different fraction of labeled vs unlabeled data.
    
    Args:
        
    dataset (:obj:`dataset.DatasetDict`): Dictionary containing training and validation datasets.
    
    labeled_frac (:obj:`float`): Fraction of training dataset to be kept as labeled dataset and 
    rest will be divided as unlabeled dataset.

    model_type (:obj:`str`): Semi supervised model type.

    labeled1_frac (:obj:`float`): In the case of CoTraining and TriTraining model_type, this 
    is the fraction given to the first two models (m1 and m2) after being divided by labeled_fr. 
    Rest is given to model 3. For example, labeled1_frac=0.33, m1 and m2 gets 0.33 
    and m3 gets (1-2*0.33) 
    
    train_key (:obj:`str`):  Key value of where training data is accessed.

    label_column (:obj:`str`):  Key value of where columns for labels.

    unlabeled_labels (:obj:`int`):  Value to be assigned to the unlabeled dataset labels, 
    required for Pi, TemporalEnsemble, and MeanTeacher as they need to know which ones 
    are unlabeled examples.
    
    batchsize (:obj:`int`):  Batch size used during training.

    Return:
    dataset.DatasetDict object with labeled and unlabeled data. 
    """

    
    dataset_train = dataset[train_key].shuffle()
    num_rows = dataset_train.num_rows
    
    lab_lim = match_with_batchsize(int(labeled_fr*num_rows), batchsize)
    unlab_start = num_rows - match_with_batchsize(num_rows-lab_lim, batchsize)
    dataset['labeled'] = dataset_train.select(np.arange(0, lab_lim))        
    dataset['unlabeled'] = dataset_train.select(np.arange(unlab_start, num_rows))
    
    if model_type=='PiModel' or model_type=='TemporalEnsemble' or model_type == 'MeanTeacher':
        
        dataset['unlabeled'] = dataset['unlabeled'].map(lambda x:{label_column:unlabeled_labels})
        
        del dataset[train_key]
        
        dataset[train_key] = concatenate_datasets([dataset['labeled'], dataset['unlabeled']])
        
        del dataset['unlabeled']
        gc.collect()
        return DatasetDict(dataset)
    
    elif model_type=='CoTrain':
        
        dataset['labeled1'] = dataset_train.select(np.arange(0, int(labeled1_frac*lab_lim)))
        dataset['labeled2'] = dataset_train.select(np.arange(int(labeled1_frac*lab_lim), lab_lim))
        dataset['unlabeled'].remove_columns_(label_column)

    elif model_type=='TriTrain':
        
        dataset['labeled1'] = dataset_train.select(np.arange(0, int(labeled1_frac*lab_lim)))
        dataset['labeled2'] = dataset_train.select(np.arange(int(labeled1_frac*lab_lim), int(2*labeled1_frac*lab_lim)))
        dataset['labeled3'] = dataset_train.select(np.arange(int(2*labeled1_frac*lab_lim), lab_lim))
        dataset['unlabeled'].remove_columns_(label_column)
    
    else:
        
        dataset['labeled'] = dataset_train.select(np.arange(0, lab_lim))        
        dataset['unlabeled'].remove_columns_(label_column)
    
    del dataset[train_key]
    gc.collect()
    return DatasetDict(dataset)

def dic_to_pandas(history, loss_key='eval_loss', accuracy_measure='eval_matthews_correlation'):
    """ 
    Function to convert the list of dictionary to pandas DataFrame which is easier for the plotting
    function in plotting utils to handle
    
    Args:
    
    history (:obj:`list`): A list of history dictionaries. It is basically 
    transformer.TrainerState() at different hyperparameters analysed. 
    
    loss_key (:obj:`str`): The key to look for. In the case of analysis of evaluation history, 
    the key is 'eval_loss'. In the case of analysis of training history the key is 'train_loss'
   
    accuracy_measure (:obj:`str`): Name of the metric used during evaluation.
    
    """

    enteries_in_old_dics=0
    dic_state=0
    
    stats = defaultdict(list)
    
    for state in history:
        
        for dics in state:
            
            if type(dics)==dict and loss_key in dics.keys():
                
                stats['epoch'].append(dics["epoch"])
                stats['step'].append(dics["step"])
                stats[loss_key].append(dics[loss_key])
                
                if accuracy_measure: stats[accuracy_measure].append(dics[accuracy_measure])
                
                enteries_in_old_dics += 1
            
            elif type(dics)==list or type(dics)==tuple:
                for _ in range(enteries_in_old_dics - dic_state):
                    stats[dics[0]].append(dics[1])
        
        dic_state = enteries_in_old_dics
    
    return pd.DataFrame(data=stats)

class SimpleDataset(Dataset):
    """
    A simple dataset for utilities in Co-Training and Tri-Training.   

    Args:
        
    dataset (:obj:`Union[SimpleDataset, dataset]`): Can be SimpleDataset object 
    or pyarrow based dataset object.

    Class attributes:
    
     -**original_len**: Length of the dataset at the instantiation.
     
     -**to_append_dic**: The dictionary used in appending unlabeled examples in dataset.
     
     -**batch_masks**: This dictionary keeps track of the unlabeled examples which are 
     removed and inserted in the dataset during appending procedure.
     
    """
   
    def __init__(self, dataset):
        
        if isinstance(dataset, SimpleDataset):
            self.dataset = dataset.dataset
        else:
            self.dataset = dataset
        
        self.original_len = len(self.dataset)
        
        self.to_append_dic = defaultdict(list)
        self.batch_masks = defaultdict(torch.tensor)
                    
    def append(self, ul_data, mask = None, batch_index = None):

        """ 
        Function used during the appending procedure. 
        
        Args:
        
        ul_data (:obj:`torch.FloatTensor`): Unlabeled data batch.
        
        mask (:obj:`torch.BoolTensor`): Mask of the data object which are going to accepted 
        from the batch. This object also helps in keeping track of the examples which are inserted. 
        
        batch_index (:obj: ´int´): Index of the batch of unlabeled data. To be used by batch_masks
        dictionary. 
        
        Return:
        mask_change.sum() (:obj: ´int´): Sum of any insertion and deletion of examples in the dataset. 
        
        """

        mask = torch.ones(ul_data['input_ids'].size()[0]).bool() if mask is None else mask
        
        for k in ul_data:
            to_add = ul_data[k][mask].tolist()
            if len(to_add):
                self.to_append_dic[k] += to_add

        if batch_index is not None:

            mask_change = mask.clone()

            if batch_index not in self.batch_masks.keys():
                self.batch_masks[batch_index] = mask

            else:
                exists = torch.logical_and(self.batch_masks[batch_index], mask)
                mask_change[exists]=False

                changed_to_false = torch.logical_and(self.batch_masks[batch_index], mask==False)
                self.batch_masks[batch_index][changed_to_false]=False
                mask_change[changed_to_false]=True

                self.batch_masks[batch_index] = torch.logical_or(self.batch_masks[batch_index], mask)
            
            return mask_change.sum()            
        
            
    def reset(self):
        """ 
        Resets the dataset to the original length at the instantiation.                 
        """
        self.dataset = self.dataset.select(np.arange(0, self.original_len))
        
    def reformat(self):

        """ 
        After appending using usual list appending, dataset is reformat to 
        huggingface dataset format.
        """

        for k in self.to_append_dic:
            self.to_append_dic[k] = self.dataset[k][:] + self.to_append_dic[k]
        
        if len(self.to_append_dic):
            self.dataset = Dataset_hf.from_dict(self.to_append_dic)
        
        self.to_append_dic.clear()
        gc.collect()
        
        return self 
        
    def __len__(self):
    
        return len(self.dataset)
    
    def __getitem__(self, idx):
        
        return self.dataset[idx]
            
    def extend_length(self, length):
        """ 
        Extends the length of the dataset by randomly repeating length amount of rows.
        """
        
        len_=len(self)
        
        if length<len_:
            raise( 'Should not decrease the length of dataset' )
        
        rand_indices = np.random.randint(len_, size=length - len_)
        columns = self.dataset.format['columns']
        
        additional_data = self.dataset.select(rand_indices)
        self.dataset = concatenate_datasets([self.dataset, additional_data])
        self.dataset.set_format(type=self.dataset.format["type"], columns=columns)
