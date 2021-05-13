import torch
from transformers import Trainer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from .dataset_utils import modify_datasets, SimpleDataset, extract_keys
import shutil
import numpy as np
import pandas as pd
import gc
from collections import defaultdict
import itertools
import inspect
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.optimization import get_scheduler, AdamW
from torch.optim.lr_scheduler import MultiplicativeLR
from copy import deepcopy
from typing import Optional, Tuple
from torch.optim import Optimizer
from transformers.trainer_callback import TrainerState
from transformers.trainer_pt_utils import nested_detach

def get_linear_schedule_with_minlr(optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int , last_epoch: int = -1, min_lr : int = 1e-07):
    
    """
    Creates a scheduler with a learning rate that linearly decreases but saturates at min_lr value. 
    
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`int`, `optional`, defaults to 1):
            The number of hard restarts to use.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
        min_lr (:obj:`int`, `optional`, defaults to 1e-07):
            The value of minimum learning rate where it should saturate.

    Return:
        :obj:`torch.optim.lr_scheduler.MultiplicativeLR` with the appropriate schedule.
    """

    init_lr = optimizer.defaults['lr']
    
    def lr_lambda(current_step: int):
        
        steps_done = float(num_training_steps - current_step)
        
        if current_step>1:
            mul_fac = steps_done / max(steps_done+1, 1)
        else:
            mul_fac = steps_done /(num_training_steps)
        
        if mul_fac*init_lr>min_lr:
            return mul_fac  
        else:
            return 1

    return MultiplicativeLR(optimizer, lr_lambda, last_epoch)

class RemoveUnusedColumnMixing:
    
    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        
        """
        Simply replacing model.pretrained_model.forward and model.teacher.forward in the 
        case of MeanTeacher and NoisyStudent from model.forward in
        ~transformers.Trainer._remove_unused_columns.
        """        

        if not self.args.remove_unused_columns:
            return
        # Inspect model forward signature to keep only the arguments it accepts.
        if self.model.type_=='MeanTeacher' or self.model.type_=='NoisyStudent':
            signature = inspect.signature(self.model.teacher.forward)
        else:
            signature = inspect.signature(self.model.pretrained_model.forward)

        signature_columns = list(signature.parameters.keys())
        # Labels may be named label or label_ids, the default data collator handles that.
        signature_columns += ["label", "label_ids"]
        columns = [k for k in signature_columns if k in dataset.column_names]

        dataset.set_format(type=dataset.format["type"], columns=columns)


class UWScheduler:
    
    """
    Unsupervised weights scheduler for changing the unsupervised weight
    of the semi supervised learning models. Also contains methods any 
    other kinds variables updates required by the models. For example, 
    PiModel, TemporalEnsembling model, and Mean Teacher. In this implementation, 
    it's  based on the composition with learning scheduler from pytorch and it 
    works best ~transformers.Trainer without having to rewrite train method. 
    # TODO: cleaner version with the rewritten Trainer.train method.
    
    Args:
        
    lr_scheduler (:obj:`torch.optim.lr_scheduler`): Learning scheduler object. 
    
    trainer (:obj:`~TrainerWithUWScheduler`): Trainer object.
    
    unsup_start_epochs (:obj:`int`): value of epoch at which the unsupervised 
    weights should start updating.

    max_w (:obj:`float`): maximum value of weight that the unsup_weight from model 
    could reach.

    update_teacher_steps (:obj:`int`): useful for MeanTeacher, sets the interval after 
    which teacher variables should be updated.
    
    w_ramprate (:obj:`float`): linear rate at which the unsupervised weight would be 
    increased from the initial value.
    
    update_weights_steps (:obj:`int`): interval steps after which unsupervised weight 
    would be updated by the w_ramprate.
    
    Class attributes:
     
    -**step_in_epochs**: Number of steps (batch passes) in an epoch.

    -**local_step**: keeps track of the times unsupervised weight has been changed.
     
    """

    def __init__(self, lr_scheduler, trainer, unsup_start_epochs=0, max_w=1,
                 update_teacher_steps=False, w_ramprate=1, update_weights_steps=1):
        
        self.trainer = trainer
        self.lr_scheduler = lr_scheduler
        self.local_step=0
        self.max_w=max_w
        self.unsup_start_epochs = unsup_start_epochs
        self.update_weights_steps = update_weights_steps
        self.w_ramprate = w_ramprate
        self.steps_in_epoch = len(self.trainer.train_dataset)//self.trainer.args.train_batch_size
        self.update_teacher_steps = update_memory_logits if update_teacher_steps else self.steps_in_epoch

        
    def step(self):
        
        """
        Implementation of composition of the pytorch learning rate scheduler step function
        with schedule of unsupervised weights. Also implements updating the memory logits 
        for TemporalEnsembleModel and updating teacher variables for MeanTeacher model.
        
        """

        self.lr_scheduler.step()

        if self.trainer.state.epoch > self.unsup_start_epochs  and self.is_true(self.update_weights_steps):
            self.trainer.model.unsup_weight = min(self.max_w, self.w_ramprate*self.local_step)
            self.local_step += 1
            
        if self.trainer.model.type_=="TemporalEnsembleModel" and self.is_true(self.steps_in_epoch):
            self.trainer.model.update_memory_logits(int(self.trainer.state.epoch + 1))
        
        if self.trainer.model.type_=="MeanTeacher" and self.is_true(self.update_teacher_steps):
            self.trainer.model.update_teacher_variables() 
   
    def is_true(self, value):
        """
        A simple checker function to if it is time to update things depending on the value of value.
        """
        return (self.trainer.state.global_step and (self.trainer.state.global_step + 1)%value==0)
        
    def __getattr__(self, name):
        """
        Needed for the calls from ~transformers.Trainer.train() method.
        """
        return getattr(self.lr_scheduler, name)



class TrainerWithUWScheduler(RemoveUnusedColumnMixing, Trainer):
    
    """
    Subclass of ~transformers.Trainer with minimal code change and integration 
    with unsupervised weight scheduler. 
    
    Args:
        
    kwargs_uw: dictionary of arguments to be used by UWScheduler. 
    
    kwargs: dictionary arguments for the ~transformers.Trainer, of dataset used 
    by the trainer and could also include arguments of UWScheduler.
    
    Note: dataset for training can be given to the trainer in two ways.
    
    (i) dataset: In this case, it should the naming scheme of dataset_utils.modify_datasets. 
    
    (ii)train_dataset: Same naming scheme as used by ~transformers.Trainer. 
         
    """

    def __init__(self, kwargs_uw=None, *args, **kwargs):
        
        self.kwargs_uw = kwargs_uw if kwargs_uw else extract_keys(UWScheduler, kwargs)
        
        dataset = kwargs.pop('dataset', None)
        
        if dataset:
            kwargs['train_dataset'] = dataset["train"]        
        
        super().__init__(*args, **kwargs)  

        self.check_for_consistency()
        
    def create_optimizer_and_scheduler(self, num_training_steps):
        
        """
        Overriden ~transformers.Trainer.create_optimizer_and_scheduler with integration
        with the UWScheduler to its Trainer.lr_scheduler object
        
        """        
        super().create_optimizer_and_scheduler(num_training_steps)
        
        self.lr_scheduler = UWScheduler(self.lr_scheduler, self, **self.kwargs_uw)

    def get_train_dataloader(self):

        """
        Slightly changed ~transformers.Trainer.get_train_dataloader as models used in 
        Trainer do not allow for mixing of labeled and unlabeled data. So changing to 
        SequentialSampler instead of RandomSampler. 
        
        """        
        
        train_sampler = SequentialSampler(self.train_dataset)
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )
    
    def check_for_consistency(self):
        """
        Checks if the labeled and unlabeled are present in the same minibatch, raises error if they are. 
        """        
        
        labels = torch.tensor(self.train_dataset['label'])
        batch_size = self.args.per_device_train_batch_size
        
        for ind in range(len(labels)//batch_size):
            
            min_batch = labels[ind*batch_size : (ind+1)*batch_size]
            if not (all(min_batch>=0) or all(min_batch<0)):
                raise RuntimeError('Mixing of labeled and unlabeled examples is not allowed.')    
 
        
class BaseForMMTrainer(RemoveUnusedColumnMixing, Trainer):
    
    """
    Base class for all the mutimodel trainers. This class contains 
    the methods which are used by the Trainers which helps in training 
    the semi supervised way which have mutiple models .

    """
    def get_dataloader(self, dataset, sequential=False):
        
        """
        Slightly changed ~transformers.Trainer.get_train_dataloader
        with the flexibility to change between sequential and RandomSampler. 
        """        

        if sequential==True:
            sampler = SequentialSampler(dataset)
        else:
            sampler = RandomSampler(dataset)

        return DataLoader(
            dataset,
            batch_size=self.args.train_batch_size,
            sampler = sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    
    def pre_train_init(self, num_training_steps):
        
        """
        Collection of all the callback functions called before initializing 
        the training. See ~transformers.Trainer.train() for more details.
        
        Args:
        num_training_steps (:obj:`int`): total number of training which are 
        calculated by number of mini batches per epoch * number of epochs.

        """        
       
        self.state = TrainerState()
        self.create_optimizer_and_scheduler(num_training_steps)
        
        if self.use_min_lr_scheduler is not None:
            self.lr_scheduler = get_linear_schedule_with_minlr(self.optimizer, num_warmup_steps=self.args.warmup_steps,
                                                          num_training_steps=num_training_steps, min_lr=self.min_lr)
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.control = self.callback_handler.on_train_begin(self.args, self.state, self.control)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
                    
    def post_epoch(self, step, epoch, tr_loss):
        
        """
        Collection of all the callback functions called 
        after the epoch is done. See ~transformers.Trainer.train() for more details.
        
        Args:
        step (:obj:`int`): step number, number of steps passed out of 
        num_training_steps used in pre_train_init method.
        
        epoch (:obj:`int`): epoch passed.
        
        tr_loss(:obj:`float`): training loss.

        """        
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)                
        self.optimizer.step()
        self.lr_scheduler.step()
        self.model.zero_grad()
        self.state.global_step += 1
        self.state.epoch = self.global_epoch + epoch + (step + 1) / self.steps_in_epoch
        self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)
        self._maybe_log_save_evaluate(tr_loss, self.model, trial=None, epoch=epoch)

    def equate_lengths(self, model1_train, model2_train):
        
        """
        A method useful for CoTrain, TriTrain model training. It finds 
        whichever model dataset has less training examples and then 
        equates them using SimpleDataset.extend_length method

        Args:
        model1_train (:obj:`SimpleDataset`): Training dataset for model 1. 
        
        model2_train (:obj:`SimpleDataset`): Training dataset for model 2.

        """        

        if len(model1_train) > len(model2_train):
            model2_train.extend_length(len(model1_train))
        elif len(model2_train) > len(model1_train):
            model1_train.extend_length(len(model2_train))
            
    def confi_prediction(self, logits_m1, logits_m2, logits_m3=None):
        
        """
        Prediction made based on between confidence of the models. First 
        checks whichever model has the highest probability (confidence) 
        on a given example and choses that class as the final answer. 
        
        Args:
        logits_m1 (:obj:`torch.FloatTensor`): logits recieved from model 1. 

        logits_m2 (:obj:`torch.FloatTensor`): logits from model 2.

        logits_m3 (:obj:`torch.FloatTensor`): logits from model 3.

        """        
        
        p_label1, labels1 = torch.max(logits_m1, 1)
        p_label2, labels2 = torch.max(logits_m2, 1)
        
        logits_out = torch.zeros(logits_m1.size(), device=logits_m1.device)
                
        if logits_m3 is None:

            m1_confi = p_label1>=p_label2
            m2_confi = p_label2>=p_label1

        else:
            p_label3, labels3 = torch.max(logits_m3, 1)
            m1_confi = torch.logical_and(p_label1>=p_label2, p_label1>=p_label3)
            m2_confi = torch.logical_and(p_label2>=p_label1, p_label2>=p_label3)
            m3_confi = torch.logical_and(p_label3>=p_label1, p_label3>=p_label2)
            logits_out[m3_confi, :] = logits_m3[m3_confi, :]

        logits_out[m1_confi, :] = logits_m1[m1_confi, :]
        logits_out[m2_confi, :] = logits_m2[m2_confi, :]
        
        return logits_out
           
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys= None,
    ):
        
        """
        Slightly changed ~transformers.Trainer.prediction_step using 
        confi_prediction method to find prediction of Cotrain and TriTrain Models.
        Used during evaluation step.
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None and "labels" in inputs:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
                if isinstance(outputs, dict):
                    logits = {k:v for k, v in outputs.items() if k not in ignore_keys + ["loss"]}
                    logits = self.confi_prediction(**logits)
                else:
                    
                    logits = outputs[1:]
            else:
                loss = None
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index if has_labels else self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        return (loss, logits, labels)
    
        

class TrainerForCoTraining(BaseForMMTrainer):
    
    """
    Subclass of ~transformers.Trainer with changes for Cotrain Model.

    Args:
    
    epoch_per_cotrain (:obj:`int`, `optional`, defaults to 2): Number of epochs to 
    pass through training data while going through one iteration of cotraining.
    
    exchange_threshold (:obj:`int`, `optional`, defaults to 20): Threshold value 
    of exchange between the model, below which co training is stopped.

    ntimes_before_saturation (:obj:`int`, `optional`, defaults to 2): In the case of 
    get_linear_schedule_with_minlr, number of times should go over all model dataset 
    + unlabeled dataset before saturating. 
    
    p_threshold (:obj:`float`): threshold probability for considering exchange between models.
    
    use_min_lr_schedule (:obj:`bool`): Whether to use linear_schedule_with_minlr.
    
    min_lr (:obj:`int`, `optional`, defaults to 1e-07): The value of minimum learning rate 
    used by linear_schedule_with_minlr.
    
    show_exchange (:obj:`bool`): Whether to print the exchange happening between models.
    
    max_passes (:obj:`int`): Maximum number of passes through the all the datasets.


    Note: dataset for training can be given to the trainer in two ways.
    
    (i) dataset: In this case, it should the naming scheme of dataset_utils.modify_datasets. 
    
    (ii)dataset_model1, dataset_model2, unlabeled: dataset for model1, model2 and 
    unlabeled dataset respectively.
    
    
    Class attributes:
     
    -**total_dataset_len**: total length of all datasets including model1, model2 and unlabeled.

    -**global_epoch**: number of epoch globally including the epochs over 
    model1_dataset+unlabeled and model2_dataset +unlabeled.
         
    """
 
    def __init__(self,  epoch_per_cotrain=2, exchange_threshold=20, ntimes_before_saturation=2,  p_threshold=0.65, use_min_lr_scheduler=True, min_lr=1e-07, show_exchange=True, max_passes=1, *args, **kwargs):
        
        dataset = kwargs.pop('dataset', None)
        
        self.dataset_model1 = dataset['labeled1'] if dataset else kwargs.pop("dataset_model1")
        self.dataset_model2 = dataset['labeled2'] if dataset else kwargs.pop("dataset_model2")
        self.dataset_unlabeled = dataset['unlabeled'] if dataset else kwargs.pop("unlabeled")
        self.use_min_lr_scheduler=use_min_lr_scheduler
        ##needed ~transformers.Trainer's constructor to not to initate the lr_scheduler. 
        self.lr_scheduler = use_min_lr_scheduler
        
        super().__init__(*args, **kwargs)
        
        self.min_lr = min_lr
        self.epoch_per_cotrain = epoch_per_cotrain
        self.exchange_threshold = exchange_threshold
        self.ntimes_before_saturation = ntimes_before_saturation
        self.p_threshold = p_threshold
        self.show_exchange = show_exchange
        self.max_passes = max_passes
        
        self._remove_unused_columns(self.dataset_unlabeled, description="unlabeled")
        self._remove_unused_columns(self.dataset_model1, description="labeled1")
        self._remove_unused_columns(self.dataset_model2, description="labeled2")
        self.dataset_model1 = SimpleDataset(self.dataset_model1)
        self.dataset_model2 = SimpleDataset(self.dataset_model2)
        self.dataset_unlabeled = SimpleDataset(self.dataset_unlabeled)
        
        self.total_dataset_len = (len(self.dataset_model1) + 
                                  len(self.dataset_model2) +
                                  len(self.dataset_unlabeled))
        self.global_epoch=0
        

    def train(self):  
        """
        Train method for CoTrainer. Performs exchange of unlabeled dataset 
        between model1 and model2 until exchange condition is true, see 
        exchange_unlabeled_data(). 
        """
        
        num_training_steps = int(self.ntimes_before_saturation * self.total_dataset_len * 
                                 self.epoch_per_cotrain/self.args.train_batch_size)        
        
        self.pre_train_init(num_training_steps)
        self.cotrain(self.dataset_model1, self.dataset_model2)
        
        exchange=True
        count_passes=0
        while exchange==True and count_passes<self.max_passes:
            ul_dataloader = self.get_dataloader(self.dataset_unlabeled)
            with torch.no_grad():
                exchange = self.exchange_unlabeled_data(ul_dataloader)
            
            if exchange : self.cotrain(self.dataset_model1, self.dataset_model2)

            count_passes += 1
            

    def cotrain(self, model1_train, model2_train):
        """
        Method where the real co training takes place. Firstly, the dataset 
        for the both models are equated with randomly repeated examples from 
        training dataset, see SimpleDataset.extend_length(). Then, a batch from 
        both model1 and model2 dataset is passed to cotrain_forward, where applies 
        the orthogonality based between the classifier layers and calculates the 
        total loss from both the batches.
        
        Args:
        model1_train (:obj:`SimpleDataset`): Training dataset for model 1 after
        the exchange_unlabeled_data has been used. 
        
        model2_train (:obj:`SimpleDataset`): Training dataset for model 2 after 
        the exchange_unlabeled_data has been used.

        """

        self.equate_lengths(model1_train, model2_train)
        model1_dataloader = self.get_dataloader(model1_train)
        model2_dataloader = self.get_dataloader(model2_train)            
        self.steps_in_epoch = len(model1_dataloader)
        
        self.model.zero_grad()
        
        for epoch in range(self.epoch_per_cotrain):
            
            self.control = self.callback_handler.on_epoch_begin(self.args, self.state, self.control)
            
            for step, (batch_for_model1, batch_for_model2) in enumerate(zip(model1_dataloader, model2_dataloader)):
                
                self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control)

                self.model.train()
                batch_for_model1 = self._prepare_inputs(batch_for_model1)
                batch_for_model2 = self._prepare_inputs(batch_for_model2)
                
                outputs = self.model.cotrain_forward(batch_for_model1, batch_for_model2)
                tr_loss = outputs.loss
                tr_loss.backward()
                self.post_epoch(step, epoch, tr_loss)
                
            self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)

            if self.control.should_evaluate: self.evaluate()
            
        self.global_epoch += self.epoch_per_cotrain    
    
            
    def exchange_unlabeled_data(self, ul_dataloader):
        
        """
        Method to exchange the unlabeled dataset between models. Examples on which
        model1 is confident on(above p_threshold) are given to model2 to train on and vice versa. 
        
        Args:
        ul_dataloader (class:`~torch.utils.data.DataLoader`): dataloader for unlabeled dataset.
        
        Returns:
        exchange (:obj:`bool`): If the number of the examples exchanged are above the 
        exchange_threshold.
        
        dataset_model1, dataset_model2 (:obj:`SimpleDataset`): dataset for model1 and 
        model2 with added unlabeled data. 
        
        """
        
        self.dataset_model1.reset()
        self.dataset_model2.reset()
        added = 0
        
        for batch_index, inputs in enumerate(ul_dataloader):
            
            inputs = self._prepare_inputs(inputs)

            logits1 = self.model.simple_forward(**inputs, classifier_num=0)
            logits2 = self.model.simple_forward(**inputs, classifier_num=1)
            
            p_label1, labels1 = torch.max(logits1, 1)
            p_label2, labels2 = torch.max(logits2, 1)
            
            ##labels on which model 1 is confident on.
            labels_m1_confi = torch.logical_and(p_label1>=self.p_threshold, p_label2<self.p_threshold)
            labels_m1_confi = torch.logical_and(labels_m1_confi, labels1!=labels2)
            
            ##labels on which model 2 is confident on.
            labels_m2_confi = torch.logical_and(p_label2>=self.p_threshold, p_label1<self.p_threshold)
            labels_m2_confi = torch.logical_and(labels_m2_confi, labels1!=labels2)
            
            inputs['label'] = labels2
            added += self.dataset_model1.append(inputs, labels_m2_confi, batch_index)
            
            inputs['label'] = labels1
            added += self.dataset_model2.append(inputs, labels_m1_confi, batch_index)
        
        exchange = True if added>self.exchange_threshold else False

        if self.show_exchange: print(f'Number of unlabeled examples exchanged: {added.item()}')

        self.dataset_model1 = self.dataset_model1.reformat()
        self.dataset_model2 = self.dataset_model2.reformat()

        return exchange
        

class TrainerForTriTraining(TrainerForCoTraining):
    
    """
    Subclass of ~TrainerForCoTraining with adding of a third model for Tritrain Model.

    Args:
    
    procedure (:obj:`str`, `optional`, defaults to 'agreement'): Whether 
    to train with TriTraining with agreement <https://ieeexplore.ieee.org/document/1512038> 
    or with disagreement <https://www.aclweb.org/anthology/P10-2038/>.

    Note: dataset for training can be given to Trainer as same way as TrainerForCoTraining 
    but with and addition of the third dataset for model3.
    
    (i)dataset: In this case, it should the naming scheme of dataset_utils.modify_datasets. 
    
    (ii)dataset_model3: dataset for model3.
    
    """

    def __init__(self, procedure='agreement', epoch_per_tritrain=2, *args, **kwargs):
        
        
        dataset_bool = True if 'dataset' in kwargs.keys() else False
        
        self.dataset_model3 = kwargs['dataset']['labeled3'] if dataset_bool else kwargs.pop("dataset_model3")
        
        super().__init__(*args, **kwargs)
        
        if procedure=='agreement':
            self.exchange_proc = self.agreement_proc
        else:
            self.exchange_proc = self.disagreement_proc

        self.epoch_per_tritrain = epoch_per_tritrain

        self.total_dataset_len = (len(self.dataset_model1) + 
                                  len(self.dataset_model2) + 
                                  len(self.dataset_model3) + 
                                  len(self.dataset_unlabeled))
    
        self._remove_unused_columns(self.dataset_model3, description="labeled3")
        self.dataset_model3 = SimpleDataset(self.dataset_model3)
        
    def train(self):
        """
        Train method for TriTrainer. Performs exchange of unlabeled dataset between
        three models, model1, model2 and model3 until exchange condition is true, 
        see exchange_unlabeled_data(). 
        """
        
        num_training_steps = int(self.ntimes_before_saturation * self.total_dataset_len * 
                                 self.epoch_per_cotrain/self.args.train_batch_size)
        
        
        self.pre_train_init(num_training_steps)
        
        self.tri_train(self.dataset_model1, self.dataset_model2, self.dataset_model3)

        exchange=True 
        count_passes=0
        while exchange==True or count_passes<self.max_passes:
            ul_dataloader = self.get_dataloader(self.dataset_unlabeled, True)
            
            with torch.no_grad():
                exchange = self.exchange_unlabeled_data(ul_dataloader)
                
            if exchange : self.tri_train(self.dataset_model1, self.dataset_model2, self.dataset_model3)
            count_passes += 1


    def tri_train(self, model1_train, model2_train, model3_train):
        
        """
        Method for tri training. Same procedure as TrainerForCoTraining.co_train 
        but with the addition of training for the third model.
        """        
        
        self.equate_lengths(model1_train, model2_train)
        self.equate_lengths(model2_train, model3_train)
        
        self.model.zero_grad()
        
        model1_dataloader = self.get_dataloader(model1_train)
        model2_dataloader = self.get_dataloader(model2_train)
        model3_dataloader = self.get_dataloader(model3_train)
        self.steps_in_epoch = len(model3_dataloader)
        
        for epoch in range(self.epoch_per_tritrain):            
            
            self.control = self.callback_handler.on_epoch_begin(self.args, self.state, self.control)
            
            for step, batches in enumerate(zip(model1_dataloader, model2_dataloader, model3_dataloader)):
                
                self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control)

                self.model.train()

                batch_for_model1 = self._prepare_inputs(batches[0])                
                batch_for_model2 = self._prepare_inputs(batches[1])                
                batch_for_model3 = self._prepare_inputs(batches[2])
                
                ct_outputs = self.model.cotrain_forward(batch_for_model1, batch_for_model2)
                m3_output = self.model.m3_forward(**batch_for_model3)
                tr_loss = ct_outputs.loss + m3_output.loss
                tr_loss.backward()
                
                self.post_epoch(step, epoch, tr_loss)
                
            self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)
            
            if self.control.should_evaluate: self.evaluate()
        
        self.global_epoch += self.epoch_per_tritrain
    
    def disagreement_proc(self, labels_confi, la, lb, l_compare):
        """
        Function to implement disagreement procedure during the exchange of unlabeled data.
        
        """        
        labels_confi = torch.logical_and(labels_confi, la==lb)
        labels_confi = torch.logical_and(labels_confi, la!=l_compare)
        return torch.logical_and(labels_confi, lb!=l_compare)
            
    def agreement_proc(self, labels_confi, la, lb, l_compare):        
        """
        Function to implement agreement procedure during the exchange of unlabeled data.
        
        """        
        return torch.logical_and(labels_confi, la==lb)
    
    
    def exchange_unlabeled_data(self, ul_dataloader):
        
        """
        Method to exchange the unlabeled dataset between models. Based on 
        chosen exchange procedure. Rest is similar to TrainerForCoTraining.exchange_unlabeled_data().
        
        Args:
        ul_dataloader (class:`~torch.utils.data.DataLoader`): dataloader for unlabeled dataset.
        
        Returns:
        exchange (:obj:`bool`): If the number of the examples exchanged are above the exchange_threshold.
        """
 
        self.dataset_model1.reset()
        self.dataset_model2.reset()
        self.dataset_model3.reset()

        added=0
        for batch_index, inputs in enumerate(ul_dataloader):
            
            inputs = self._prepare_inputs(inputs)

            logits1 = self.model.simple_forward(**inputs, classifier_num=0)
            logits2 = self.model.simple_forward(**inputs, classifier_num=1)
            logits3 = self.model.simple_forward(**inputs, classifier_num=2)
            
            p_label1, labels1 = torch.max(logits1, 1)
            p_label2, labels2 = torch.max(logits2, 1)
            p_label3, labels3 = torch.max(logits3, 1)
            
            labels_m1m2_confi = torch.logical_and(p_label1>=self.p_threshold, p_label2>=self.p_threshold)
            labels_m1m2_confi = self.exchange_proc(labels_m1m2_confi, labels1, labels2, labels3)
            
            labels_m2m3_confi = torch.logical_and(p_label2>=self.p_threshold, p_label3>=self.p_threshold)
            labels_m2m3_confi = self.exchange_proc(labels_m2m3_confi, labels2, labels3, labels1)

            labels_m1m3_confi = torch.logical_and(p_label1>=self.p_threshold, p_label3>=self.p_threshold)
            labels_m1m3_confi = self.exchange_proc(labels_m1m3_confi, labels1, labels3, labels2)

            inputs['label'] = labels2
            added += self.dataset_model1.append(inputs, labels_m2m3_confi, batch_index)

            inputs['label'] = labels1
            added += self.dataset_model2.append(inputs, labels_m1m3_confi, batch_index)

            inputs['label'] = labels2
            added += self.dataset_model3.append(inputs, labels_m1m2_confi, batch_index)

        exchange = True if added>self.exchange_threshold else False
        
        if self.show_exchange: print(f'Number of unlabeled examples exchanged: {added.item()}')
        
        self.dataset_model1 = self.dataset_model1.reformat()
        self.dataset_model2 = self.dataset_model2.reformat()
        self.dataset_model3 = self.dataset_model3.reformat()
        
        return exchange
    
    
class TrainerForNoisyStudent(BaseForMMTrainer):
    
    """
    Subclass of ~transformers.Trainer for the noisy student model.

    Args:
    
    min_lr (:obj:`int`, `optional`, defaults to 1e-07): The value of minimum 
    learning rate used by get_linear_schedule_with_minlr.
    
    epoch_per_ts_iter (:obj:`int`, `optional`, defaults to 1): Number of epochs 
    to pass during each teacher student iteration.
    
    ts_iter (:obj:`int`, `optional`, defaults to 3): Number of teacher student 
    iterations during training, in which student is again used as the teacher. 

    ntimes_before_saturation (:obj:`int`, `optional`, defaults to 2): In the case 
    of get_linear_schedule_with_minlr, number of times should go over all model 
    dataset + unlabeled dataset before saturating. 

    Note: dataset for training can be given to Trainer in two ways.
    
    (i)dataset: In this case, it should the naming scheme of dataset_utils.modify_datasets. 
    
    (ii)dataset_labeled, dataset_unlabeled: dataset for labeled and unlabeled data.
    
    """

    def __init__(self, min_lr=1e-07, epoch_per_ts_iter=1, ts_iter=3, ntimes_before_saturation=2, reduce_init_lr_factor=1, use_min_lr_scheduler=None, *args, **kwargs):
        
        dataset = kwargs.pop('dataset', None)
        
        self.use_min_lr_scheduler = use_min_lr_scheduler
        self.dataset_labeled = dataset['labeled'] if dataset else kwargs.pop("dataset_labeled")
        self.dataset_unlabeled = dataset['unlabeled'] if dataset else kwargs.pop("dataset_unlabeled")
        self.lr_scheduler = use_min_lr_scheduler
        
        super().__init__(*args, **kwargs)
        
        self._remove_unused_columns(self.dataset_unlabeled, description="unlabeled")
        self._remove_unused_columns(self.dataset_labeled, description="labeled")
        self.dataset_labeled = SimpleDataset(self.dataset_labeled)
        self.dataset_unlabeled = SimpleDataset(self.dataset_unlabeled)
        
        self.min_lr = min_lr
        self.epoch_per_ts_iter = epoch_per_ts_iter
        self.ts_iter = ts_iter
        self.ntimes_before_saturation = ntimes_before_saturation
        self.reduce_init_lr_factor = reduce_init_lr_factor
        self.total_dataset_len = (len(self.dataset_labeled) + 
                                  len(self.dataset_unlabeled))

        self.global_epoch=0
        ##repoint to transformers.Trainer.prediction_step() method for this case.
        self.prediction_step = super(BaseForMMTrainer, self).prediction_step
    
    def train(self):
        """
        Train method for Noisy student. Trains the teacher and student 
        with exchange (replacing teacher with the student) at the end of 
        every iteration. In this case both optimizer and learning rate scheduler 
        at reinitiated at the end of an iteration. 
        
        """
        
        for _ in range(self.ts_iter):
            self.dataset_labeled.reset()
            self.train_and_reset(self.model.teacher)
            
            self.psuedo_label()
            
            self.train_and_reset(self.model.student)
            self.exchange_models()
            self.args.learning_rate /= self.reduce_init_lr_factor


    def train_and_reset(self, model):
        """
        Trains either teacher or student then resets the training 
        variables like optimizer and scheduler. 
        """

        self.pre_train_init(self.num_training_steps_)
        self.train_one_model(model)
        
        del self.optimizer, self.lr_scheduler
        self.lr_scheduler = self.use_min_lr_scheduler
        self.optimizer = None

    @property
    def num_training_steps_(self):
        return int(len(self.dataset_labeled)*self.epoch_per_ts_iter/self.args.train_batch_size)

    def train_one_model(self, model):        
        """
        Common method for training either the teacher or student model.
        """

        train_dataloader = self.get_dataloader(self.dataset_labeled)
        self.model.zero_grad()
        self.steps_in_epoch = len(train_dataloader)
        
        for epoch in range(self.epoch_per_ts_iter):            
            
            self.control = self.callback_handler.on_epoch_begin(self.args, self.state, self.control)
            
            for step, inputs in enumerate(train_dataloader):
                
                self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control)
                self.model.train()
                
                inputs = self._prepare_inputs(inputs)                
                tr_loss = self.model(**inputs).loss
                tr_loss.backward()
                
                self.post_epoch(step, epoch, tr_loss)
                
            self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)
            if self.control.should_evaluate: self.evaluate()

            
        self.global_epoch += self.epoch_per_ts_iter
            


    def psuedo_label(self):        
        """
        Method for generating psuedo_label for by teacher model for student.
        """
        
        ul_dataloader = self.get_dataloader(self.dataset_unlabeled)
        
        for inputs in ul_dataloader:
            
            inputs = self._prepare_inputs(inputs)
            logits = self.model.teacher(**inputs).logits
            _, labels = torch.max(logits, 1)
            inputs['label'] = labels
            self.dataset_labeled.append(inputs)            

        self.dataset_labeled = self.dataset_labeled.reformat()
            
    def exchange_models(self):
        """
        Method for changing student model into teacher model. 
        """
        
        del self.model.teacher
        gc.collect()
        
        self.model.teacher = deepcopy(self.model.student)
        self.model.teacher.dropout.p = self.model.teacher_dropout
