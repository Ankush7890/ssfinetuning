import torch
from transformers import Trainer
from transformers import AutoModelForSequenceClassification,  Trainer
from .dataset_utils import modify_datasets, SimpleDataset, extract_keys
import os, shutil
import numpy as np
import gc
from collections import defaultdict, OrderedDict
import itertools
import inspect
from .default_args import get_default_ta_sup, get_default_ta, set_default_args
from functools import wraps

from .trainer_util import (
    TrainerWithUWScheduler, 
    TrainerForCoTraining, 
    TrainerForTriTraining,
    TrainerForNoisyStudent
)
from .models import (
    PiModel,
    TemporalEnsembleModel,
    CoTrain,
    TriTrain,
    MeanTeacher,
    NoisyStudent,
)

MAPPING_TO_MODEL = OrderedDict(
    [
        # mapping to ssl models
        ("PiModel", PiModel),
        ("TemporalEnsemble", TemporalEnsembleModel),
        ("CoTrain", CoTrain),
        ("TriTrain", TriTrain),
        ("MeanTeacher", MeanTeacher),
        ("NoisyStudent", NoisyStudent),
    ]
)

MAPPING_TO_TRAINER = OrderedDict(
    [
        # mapping to ssl trainer
        ("PiModel", TrainerWithUWScheduler),
        ("TemporalEnsemble", TrainerWithUWScheduler),
        ("CoTrain", TrainerForCoTraining),
        ("TriTrain", TrainerForTriTraining),
        ("MeanTeacher", TrainerWithUWScheduler),
        ("NoisyStudent", TrainerForNoisyStudent),
    ]
)

def generate_kwargs(hyperparam_dic):
    
    """
    Generator function for all combinations of hyperparameters from hyperparameter dictionary 
    
    Args:
    
    hyperparam_dic (:obj:`Dict` ): Hyperparameter dictionary.
    
    """
    if not hyperparam_dic:
        yield {}
        return
    
    numkeys = len(hyperparam_dic.keys())
    flatten_key_vals = ([[kys, vals] for kys, val_list in hyperparam_dic.items() for vals in val_list])

    for combs in itertools.combinations(np.arange(len(flatten_key_vals)), numkeys):
        
        kys=np.array(flatten_key_vals)[list(combs)][:,0]
        
        if len(set(kys))==len(kys):
            kwargs = {flatten_key_vals[i][0] : flatten_key_vals[i][1] for i in combs}
            yield kwargs
            
        else:
            continue

def check_and_replace(key, kwargs, args, basefunction):
    
    """
    Function to check for the "key" in "kwargs" or "args" and replace args 
    if key is found in args. In case it's not found either kwargs or args, 
    it takes the default option from basefunction. 
    
    Args:
    
    key (:obj:`str`): key to be searched.
    
    kwargs (:obj:`dict`): keyword argument dictionary to look through.
    
    args (:obj:`tuple`): arguments tuple to look through.
    
    basefunction(:obj: `train_with_ssl`): Base function around which wrapper 
    has been implemented.
    
    """

    if key in kwargs.keys():
        return kwargs.pop(key), args

    elif len(args):
        val = args[0]
        args_n = args[1:]
        del args
        gc.collect()
        return val, args_n
    
    else:
        signature = inspect.signature(basefunction)
        val = signature.parameters[key].default
        
        return val, args

def with_labeled_fraction(basefunction, labeled_fraction, *args, **kwargs):
    
    """
    Wrapper function around train_with_ssl implemented if a list of labeled_fractions is mentioned.
    
    Args:
    
    basefunction(:obj: `train_with_ssl`): Base function around which wrapper has been implemented.
    
    labeled_fraction (:obj:`list`): List of the labeled fraction of training data to be analysed. 
    This function uses ~dataset_utils.modify_datasets to divide the dataset into the fraction of 
    labeled dataset and unlabeled dataset. Then, each l_fr mentioned in labeled_fraction is analysed 
    seperately and results are stored in sup_stats and stats.
    
    kwargs: Remaining dictionary arguments for the train_with_ssl function.
    
    """

    sup_stats=[]
    stats=[]
    kwargs_defs = extract_keys(set_default_args, kwargs)
        
    dataset, args = check_and_replace('dataset', kwargs_defs, args, basefunction)
    model_name, args = check_and_replace('model_name', kwargs_defs, args, basefunction)
    ssl_model_type, args = check_and_replace('ssl_model_type', kwargs, args, basefunction)
    
    dataset, kwargs_ta = set_default_args(dataset, model_name, kwargs)
    
    batch_size = kwargs_ta['per_device_train_batch_size']
    
    for l_fr in labeled_fraction:
        
        dataset_new = modify_datasets(dataset.copy(), labeled_fr = l_fr, batchsize = batch_size,
                                      model_type = ssl_model_type)
        
        basefunction(dataset_new,
             model_name,
             ssl_model_type,
             *args, 
             sup_stats = sup_stats, 
             stats = stats, 
             l_fr=l_fr,
             **kwargs)
        
        del dataset_new
        gc.collect()
    
    return sup_stats, stats        

def wrapper_for_l_fr(func):
    
    @wraps(func)
    def decorator(*args, **kwargs):
                
        if 'labeled_fraction' in kwargs.keys() or (len(args) and isinstance(args[0], list)) :
            return with_labeled_fraction(func, *args, **kwargs)
        else:
            return func(*args, **kwargs)

    return decorator


@wrapper_for_l_fr
def train_with_ssl(dataset=None,
                    model_name="distilbert-base-uncased", 
                    ssl_model_type='PiModel',
                    text_column_name='sentence',
                    run_sup = False,
                    use_sup = False,
                    remove_dirs=True,
                    teacher_student_name = None,
                    num_labels = 2,
                    unsup_hp = {'w_ramprate':[0.001,0.01,0.1], 'alpha':[0.3,0.6,0.9]},
                    sup_stats = None,
                    stats = None,
                    l_fr=False,
                    **kwargs
                    ):
    
    """
    Function for training with semisupervised models during finetuning of the pretrained 
    transformer models. 
    
    Args:
    
    labeled_fraction (:obj:`list` ): Set up by wrapper_for_l_fr. List of labeled fraction 
    of training data to be analysed. In this case original dataset is divided into the 
    fraction of labeled dataset and unlabeled dataset.

    dataset (:obj:`~datasets.DatasetDict` ): Dataset dictionary containing labeled and 
    unlabeled data.
    
    model_name (:obj:`str` or :obj:`os.PathLike`): "pretrained_model_name_or_path" in 
    ~transformers.PreTrainedModel, please refer to its documentation for further information. 
    
    (i) In this case of a string, the `model id` of a pretrained model hosted inside a model 
    repo on huggingface.co.
    
    (ii) It could also be address of saved pretrained model.
    
    ssl_model_type (:obj:`str`): Semisupervised model type. 
    
    text_column_name (:obj:`str`): Column name for the text in the dataset.
    
    run_sup (:obj:`bool`): Whether to run a supervised model along with the semi supervised model
    for comparison.
    
    use_sup (:obj:`bool`): Whether to use the trained supervised as the starting point 
    the ssl model. If this is True, run_sup has to be true too. 
    
    remove_dirs (:obj:`bool`): Whether to remove dirs created by ~transformers.Trainer.

    teacher_student_name (:obj:`Tuple[`str`, `str`]): A Tuple for teacher and student name, 
    respectively for multi transformer models like MeanTeacher or NoisyStudent. Similar to model_name.
    
    num_labels (:obj:`int`): Total number of classes for the classification.
    
    unsup_hp (:obj:`dict`): The dictionary of all the hyperparameters in unsupervised part of model. 
    Check the documentation of ssl_model_type and the associated trainers before setting up this 
    dictionary. The train_with_ssl with then train the model on all the combinations of the 
    hyperparameters set in this dictionary.
    
    sup_stats (:obj:`list`): Used by the wrapper as an argument. List to save the supervised models 
    stats for comparison, if the run_sup is turned on. If labeled_fraction is not mentioned it is 
    created within this function.
    
    stats (:obj:`list`): Used by the wrapper as an argument. List to save the chosen semisupervised
    models (ssl_model_type) stats. Similar to sup_stats.
    
    l_fr(:obj:`float`): Used by the wrapper as an argument. Float value of fraction used as 
    labeled dataset.
    
    kwargs: Remaining dictionary arguments for the transformer.Trainer init function. Some of the 
    Trainer keyword are important for training like compute metrics, and tokenizer, 
    (see ~transformer.Trainer). If they are not mentioned, the default values would be picked,
    see default_args.set_default_args().
    
    Note: ~transformer.TrainingArgument which ~transformer.Trainer accepts as the args, here could 
    be given in same way. To distinguish arguments for the supervised trainer, it should be named 
    args_ta_sup for the supervised trainer and args_ta for the semisupervised trainer. There are 
    some default keys set in default_args file. If one just needs to change only some args and keep 
    rest of them as same default, then you can also set args_ta or args_ta_sup as a dictionary.
    
    For example setting:
    args_ta_sup = {learning_rate: 1} will only change learning rate to 1. Keep rest of them similar
    to what has been set in default_args. 
    
    Return:
    
    sup_stats (:obj:`list`): If used directly without the labeled_fraction and if run_sup is True 
    else returns empty list. Information of all the training history for supervised model.
    
    stats (:obj:`list`): If used directly without the labeled_fraction. Information of all the 
    training history for semi supervised model.
        
    """
    
    SSLModel = MAPPING_TO_MODEL[ssl_model_type]
        
    if not l_fr:
        
        sup_stats=[]
        stats=[]
        dataset, _ = set_default_args(dataset, model_name, kwargs)
    
    if use_sup: 
        run_sup=True
        if ssl_model_type in ['MeanTeacher', 'NoisyStudent']:
            raise TypeError( 'Supervised models cannot be used for multi semisupervised models starting points.' )
    
    args_ta_sup_in_keys = kwargs.pop('args_ta_sup', None)
    args_ta_in_keys = kwargs.pop('args_ta', None)
    
    args_ta_sup = get_default_ta_sup(f'runs/l_fr_{l_fr}') if args_ta_in_keys is None else args_ta_sup_in_keys.copy()
    
    output_dir_sup = args_ta_sup.output_dir
    
    if run_sup:
        
        model_sup = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = num_labels)
        trainer_sup = Trainer(
            model_sup,
            args_ta_sup,
            train_dataset=dataset["labeled"],
            eval_dataset=dataset["validation"],
            **kwargs
        )
        
        trainer_sup.train()
        best_model_checkpoint=trainer_sup.state.best_model_checkpoint
        trainer_sup.state.log_history.append(('l_fr', l_fr))
        sup_stats.append(trainer_sup.state.log_history)

        del trainer_sup, model_sup, args_ta_sup
        gc.collect()

    for kwargs_ssl in generate_kwargs(unsup_hp):
        logging_dir=''
        
        if l_fr: logging_dir += f'runs/l_fr_{l_fr}_'
        
        # For tensorboard.
        for k, v in kwargs_ssl.items(): logging_dir += k + '_' + str(v) + '_' 
        
        # For train_with_ssl plotter.
        kwargs_ssl_copy = kwargs_ssl.copy()
        
        # Seperate the keys for model and trainer
        kwargs_model = extract_keys(SSLModel, kwargs_ssl) 
        
        if use_sup == True:
            model_ssl = SSLModel(model_name=best_model_checkpoint, supervised_run=True,**kwargs_model)
            
        elif teacher_student_name is not None:
            model_ssl = SSLModel(teacher_student_name=teacher_student_name, **kwargs_model)
            
        else:
            model_ssl = SSLModel(model_name=model_name, **kwargs_model)
        
        args_ta = get_default_ta(logging_dir) if args_ta_in_keys is None else args_ta_in_keys.copy()
        
        output_dir = args_ta.output_dir

        trainer_ssl = MAPPING_TO_TRAINER[ssl_model_type](**kwargs_ssl, 
                                                            model=model_ssl,
                                                            args=args_ta,
                                                            dataset=dataset,
                                                            eval_dataset = dataset['validation'],
                                                            **kwargs
                                                            )
        trainer_ssl.train()
        
        
        for kys, vals in kwargs_ssl_copy.items(): trainer_ssl.state.log_history.append((kys, vals))
        
        if l_fr: trainer_ssl.state.log_history.append(('l_fr', l_fr))

        stats.append(trainer_ssl.state.log_history)
        
        del model_ssl, trainer_ssl, args_ta
        gc.collect()
    
    if remove_dirs:
        if os.path.exists(output_dir_sup): 
            shutil.rmtree(output_dir_sup)
            
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            
    if not l_fr: 
        return sup_stats, stats
