from transformers import AutoModelForSequenceClassification
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput
from torch import linalg as LA
import torch.nn.functional as F
from transformers.file_utils import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple
import gc
import inspect
import warnings
@dataclass
class CoTrainModelOutput(ModelOutput):
    
    loss: Optional[torch.FloatTensor]  = None
    logits_m1: torch.FloatTensor = None
    logits_m2: torch.FloatTensor = None

@dataclass
class TriTrainModelOutput(ModelOutput):
    
    loss: Optional[torch.FloatTensor]  = None
    logits_m1: torch.FloatTensor = None
    logits_m2: torch.FloatTensor = None
    logits_m3: torch.FloatTensor = None

def add_signature_from(base):
    
    def decorator(derived):
        
        #These keys are irrelevant for hyperparameters but create duplicate key issues. 
        ignore_keys=['ssl_model_type', 'num_models']
        
        base_params = [v for k, v in inspect.signature(base).parameters.items() if k not in ignore_keys]
        der_params = list(inspect.signature(derived).parameters.values())
        
        ##removes the *args and **kwargs from the signature
        new_params = set(der_params[:-2]+ base_params)
        derived.__signature__ = inspect.signature(derived).replace(parameters=new_params)
        return derived
    
    return decorator 

class BaseModelClass(nn.Module):
    
    """
    Base class for all model with single pretrained model, but might have multiple classifier layers.

    Args:

    model_name (:obj:`str` or :obj:`os.PathLike`): "pretrained_model_name_or_path" 
    in ~transformers.PreTrainedModel, please refer to its documentation for further information. 
    
    (i) In this case of a string, the `model id` of a pretrained model hosted inside a model 
    repo on huggingface.co.
    
    (ii) It could also be address of saved pretrained model. 
    
    supervised_run (:obj:`bool`): If the model is taken from the supervised run or not. 
    In that case transformer_model_name is the path to the saved model. 
    
    num_labels (:obj:`int`): number of labels to be classified.
    
    classifier_dropout (:obj:`float`): dropout probability of the classifier layers.
        
    num_models (:obj:`int`): number of models, i.e. number of classifier layers 
    (only set by the sub classes).
    
    ssl_model_type (:obj:`str`): semi supervised learning model type (only set by the sub classes).    
    
    """

    def __init__(self, model_name='albert-base-v2', supervised_run=False, num_labels=2, classifier_dropout=0.1,
                 num_models=1, ssl_model_type=None):
        super().__init__()

        self.type_ = ssl_model_type
        self.num_labels = num_labels
        self.num_models = num_models
        
        if supervised_run:
            self.pretrained_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
        else:
            self.pretrained_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)                
            
        if num_models>1:
            # if number of classifiers are more than one then the AutoModels classifiers is removed and
            # reinitialized here.  
            del self.pretrained_model.classifier
            gc.collect()
            
            self.config = self.pretrained_model.config
            self.hidden_size = self.pretrained_model.config.hidden_size
            self.num_labels = self.config.num_labels
            self.classifiers = nn.ModuleList([nn.Linear(self.hidden_size, self.num_labels) for _ in range(num_models)])
            self.dropouts = nn.ModuleList([nn.Dropout(classifier_dropout) for _ in range(num_models)])
            self.softmaxs = nn.ModuleList([nn.Softmax(1) for _ in range(num_models)])
            self.__init__weights()
                
    def __init__weights(self):
        
        """
        Reinitialization of __init__weights function of 'pretrained_model' for 
        extra added classifier layers.        
        
        """
        for ind in range(self.num_models):
            self.pretrained_model._init_weights(self.classifiers[ind])
    
    def simple_forward_with_prob_logits(self, classifier_num=0, **kwargs):
        """
        This function first changes the pointer of the pretrained_model to the 
        one of the classifier defined in this class. Then applies softmax to it 
        and thus converts it to probability logits.
        
        Args:
        
        classifier_num: Index of the classifier to be used.

        kwargs: Arguments from pretrained_model.forward. 
        
        Return:
        
        logits ( torch.FloatTensor): probability logits.        
        
        """
        self.pretrained_model.classifier = self.classifiers[classifier_num]
        
        outputs = self.pretrained_model(**kwargs)

        logits = self.softmaxs[classifier_num](outputs.logits)
        
        return logits

@add_signature_from(BaseModelClass)    
class PiModel(BaseModelClass):
    
    """
    Implementation of pi model from <https://arxiv.org/abs/1610.02242>.

    Args:
        
    unsup_weight (:obj:`float`): Initial value of the weight of the unsupervised 
    loss component. Its value is controlled by unsupervised weight scheduler.  
    
    kwargs: remaining dictionary of keyword arguments from the BaseModelClass.
    
    """

    def __init__(self, unsup_weight=0, *args, **kwargs):
        super().__init__(*args, **kwargs, ssl_model_type="PiModel")
        
        self.unsup_weight=unsup_weight
        self.simple_forward = self.pretrained_model.forward
        
    def forward(self, **kwargs):
        
        """ 
        Implementation of forward function calculating the 
        semi supervised loss. Mixing of the labeled and unlabeled examples 
        in a single batch is not allowed. 
        
        Args:
        
        kwargs: Arguments from pretrained_model.forward.
        
        Return:
        transformers.modeling_outputs.SequenceClassifierOutput object with the information 
        of logits and loss function.
        """

        labels=kwargs.pop('labels')
    
        z1_logits = self.simple_forward( **kwargs ).logits
        z2_logits = self.simple_forward( **kwargs ).logits
        
        sup_loss = CrossEntropyLoss()
        unsup_loss = MSELoss()
        
        tot_loss=0
        if all(labels>=0):
            tot_loss = sup_loss(z1_logits.view(-1, self.num_labels), labels.view(-1))
    
        tot_loss += self.unsup_weight*unsup_loss(z1_logits, z2_logits)
        
        return SequenceClassifierOutput(
            loss=tot_loss,
            logits=z1_logits
        )
    
@add_signature_from(BaseModelClass)    
class TemporalEnsembleModel(BaseModelClass):
    
    """
    Implementation of Temporal ensemble model as introduced in <https://arxiv.org/abs/1610.02242>

    Args:
        
    alpha (:obj:`float`): memory of the last epochs. For more info please refer 
    to <https://arxiv.org/abs/1610.02242>.

    unsup_weight (:obj:`float`): initial value of weight of the unsupervised loss
    component. After setting the initial value, its value is controlled by 
    unsupervised weight scheduler. 
    
    kwargs: remaining dictionary of keyword arguments from the BaseModelClass.
    
    Class attributes:
    
     -**mini_batch_num**: keeps track of the mini_batch_num using forward method.
     
     -**logits_batchwise**: stores the logits of each batch passed through forward method.
     
     -**firstpass**: bool variable to track if its the first pass through the forward method.
     
    """

    def __init__(self, alpha=0.5, unsup_weight=0, *args, **kwargs):
        
        super().__init__( *args, **kwargs, ssl_model_type="TemporalEnsembleModel")
        
        self.unsup_weight=unsup_weight
        self.memory_logits=[]
        self.alpha=alpha
        self.mini_batch_num=0
        self.logits_batchwise=[]
        self.firstpass=True
        self.simple_forward = self.pretrained_model.forward
    
    def forward(self, **kwargs):
        
        """ 
        Implementation of forward function calculating the semi supervised loss.
        Mixing of the labeled and unlabeled examples in a single batch is not allowed. 
        
        Args:
        
        kwargs: Arguments from pretrained_model.forward.
        
        Return:
        transformers.modeling_outputs.SequenceClassifierOutput object 
        with the information of logits and the loss function.
        
        """
        
        labels = kwargs.pop("labels")
        logits = self.simple_forward(**kwargs).logits
        
        sup_loss = CrossEntropyLoss()
        unsup_loss = MSELoss()
        
        tot_loss=0
        if all(labels>=0):
            tot_loss = sup_loss(logits.view(-1, self.num_labels), labels.view(-1))
        
        if self.firstpass==False and self.training:
            tot_loss += self.unsup_weight*unsup_loss(self.memory_logits[self.mini_batch_num], logits)
            
        elif self.training:
            ##required to free the graph
            tot_loss += 0*logits.sum()
            
        if self.training:
            self.logits_batchwise.append(logits.detach().clone())
            self.mini_batch_num += 1    
        
        
        return SequenceClassifierOutput(
            loss=tot_loss,
            logits=logits
        ) 
    
    def update_memory_logits(self, t):
        
        """
        Method for updating the memory logits with the exponential average.
        
        Args:
        t (:obj:`int`): epoch value of bias normalization.
        
        """

        if self.firstpass==True:
            
            device = self.logits_batchwise[0].device
            
            self.memory_logits=[torch.zeros(self.logits_batchwise[0].shape,device=device) 
                                for _ in range(self.mini_batch_num)]
            self.firstpass=False
            
        for ind in range(self.mini_batch_num):
            
            self.memory_logits[ind] = (self.alpha*self.memory_logits[ind] + 
                                       (1-self.alpha)*self.logits_batchwise[ind])
            
            self.memory_logits[ind] /= (1-self.alpha**(t))
        
        self.mini_batch_num=0
        self.logits_batchwise[:]=[]

@add_signature_from(BaseModelClass)                
class CoTrain(BaseModelClass):
    
    """
    Implementation of Co Training as introduced in <https://www.cs.cmu.edu/~avrim/Papers/cotrain.pdf>

    Args:
        
    o_weight (:obj:`float`): Orthogonality weight for the two classifiers (or two models).
    
    kwargs: remaining dictionary of keyword arguments from the BaseModelClass.
         
    """

    def __init__(self, o_weight=0.01, ssl_model_type="CoTrain", num_models=2, *args, **kwargs):
        
        if type(self).__name__=='CoTrain':
            if ssl_model_type != 'CoTrain' or num_models!=2:
                raise RuntimeError('ssl_model_type or num_models can only be changed through a subclass.')
                
        
        super().__init__(*args, **kwargs, num_models = num_models, ssl_model_type = ssl_model_type)
        
        self.simple_forward = super().simple_forward_with_prob_logits
        self.o_weight=o_weight
        
    def forward(self, **kwargs):
        
        """ 
        Forward function only used during the evaluation of models.
        See ~trainer_utils.TrainerForCoTraining and ~transformers.Trainer for more details.  
        
        Args:
        
        kwargs: Arguments from pretrained_model.forward.
        
        Return:
        CoTrainModelOutput object with the information of logits of both models and the 
        loss function.
        
        """
        labels = kwargs.pop("labels")
        logits1 = self.simple_forward(0, **kwargs)
        logits2 = self.simple_forward(1, **kwargs)
        
        sup_loss = CrossEntropyLoss()
        
        loss1 = sup_loss(logits1.view(-1, self.num_labels), labels.view(-1))
        loss2 = sup_loss(logits2.view(-1, self.num_labels), labels.view(-1))
        
        tot_loss= (loss1 + loss2 + 
                   self.o_weight * LA.norm(torch.matmul(self.classifiers[0].weight.t(), self.classifiers[1].weight)))
        
        return CoTrainModelOutput(
            loss=tot_loss,
            logits_m1=logits1,
            logits_m2=logits2
        )


    def cotrain_forward(self, model1_batch, model2_batch):
        
        """ 
        Forward function used during training of models. 
        See ~trainer_utils.TrainerForCoTraining for more details.  
        
        Args:
        model1_batch (:obj: torch.FloatTensor) batch for model 1.
        model2_batch (:obj: torch.FloatTensor) batch for model 2.
        
        Return:
        CoTrainModelOutput object with the information of logits of both models and the 
        loss function.
        
        """


        labels1 = model1_batch.pop('labels')
        labels2 = model2_batch.pop('labels')

        logits1 = self.simple_forward(classifier_num=0, **model1_batch)
        logits2 = self.simple_forward(classifier_num=1, **model2_batch)
        
        sup_loss = CrossEntropyLoss()
        
        loss1 = sup_loss(logits1.view(-1, self.num_labels), labels1.view(-1))
        loss2 = sup_loss(logits2.view(-1, self.num_labels), labels2.view(-1))
        
        tot_loss =(loss1 + loss2 + 
                   self.o_weight * LA.norm(torch.matmul(self.classifiers[0].weight.t(), self.classifiers[1].weight)))

        return CoTrainModelOutput(
            loss=tot_loss,
            logits_m1=logits1,
            logits_m2=logits2
        )

@add_signature_from(CoTrain)  
class TriTrain(CoTrain):
    
    """
    Implementation of Tri Training(multi task TriTrain) as introduced 
    in <https://arxiv.org/abs/1804.09530>. Note: Here the implementation 
    is at only the fine tuning. The base network is to be pretrained transformer model.

    Args:
        
    kwargs: keyword arguments are the same as is for CoTrain class, except model 
    type string and number of models(num_models) as obvious with the name.
         
    """

    def __init__(self, *args, **kwargs):
        
        super().__init__( *args, ssl_model_type="TriTrain", num_models=3, **kwargs)
        
    
    def forward(self, **kwargs):
        
        """ 
        Forward function used during evaluation of trained models.
        See ~trainer_utils.TrainerForTriTraining and ~transformers.Trainer 
        for more details.  
        
        Args:
        
        kwargs: Arguments from pretrained_model.forward.
        
        Return:
        TriTrainModelOutput object with the information of logits of both 
        models and the loss function.
        
        """

        labels = kwargs["labels"]
        ct_output = super().forward(**kwargs)
        
        logits = self.simple_forward(2, **kwargs)
        
        sup_loss = CrossEntropyLoss()

        loss =  ct_output.loss + sup_loss(logits.view(-1, self.num_labels), labels.view(-1))
        
        return TriTrainModelOutput(
            loss = loss,
            logits_m1 = ct_output.logits_m1,
            logits_m2 = ct_output.logits_m2,
            logits_m3 = logits,
        )
    
    def m3_forward(self, **kwargs):

        """ 
        Forward function for model 3.
        See ~trainer_utils.TrainerForTriTraining and ~transformers.Trainer for
        more details.  
        
        Args:
        
        kwargs: Arguments from pretrained_model.forward.
        
        Return:
        TriTrainModelOutput object with the information of logits of both 
        models and the loss function.
        
        """
        labels = kwargs.pop("labels")

        logits = self.simple_forward(2, **kwargs)

        sup_loss = CrossEntropyLoss()
                
        loss = sup_loss(logits.view(-1, self.num_labels), labels.view(-1))
        
        return SequenceClassifierOutput(
            loss = loss,
            logits = logits)

    
class BaseMultiPretrained(nn.Module):
    
    """
    Base class for all models with multiple pretrained model. 

    Args:
    
    ssl_model_type (:obj:`str`): semi supervised learning model type.

    teacher_student_name (:obj:`Tuple[`str`, `str`]): A Tuple for teacher 
    and student name, respectively. "pretrained_model_name_or_path" in 
    ~transformers.PreTrainedModel, please refer to its documentation for 
    further information. 
    
    (i) In this case of a string, the `model id` of a pretrained model 
    hosted inside a model repo on huggingface.co.
    
    (ii) It could also be address of saved pretrained model. 
        
    num_labels (:obj:`int`): number of labels to be classified.
    
    student_dropout (:obj:`float`): dropout probability of the student classifier layers.

    teacher_dropout (:obj:`float`): dropout probability of the teacher classifier layers.    
        
    """
    
    def __init__(self, teacher_student_name=("albert-base-v2","albert-base-v2"), num_labels=2, teacher_dropout=None, student_dropout=None, ssl_model_type=None):
        
        super().__init__()
        
        self.teacher_model_name = teacher_student_name[0]        
        self.student_model_name = teacher_student_name[1]

        self.type_ = ssl_model_type
        self.num_labels = num_labels
        
        self.teacher = AutoModelForSequenceClassification.from_pretrained(self.teacher_model_name, num_labels = num_labels)
        
        if teacher_dropout is not None: self.teacher.dropout.p = teacher_dropout
        
        self.student = AutoModelForSequenceClassification.from_pretrained(self.student_model_name, num_labels = num_labels)
        
        if student_dropout is not None: self.student.dropout.p = student_dropout
    
    
            
@add_signature_from(BaseMultiPretrained)                
class MeanTeacher(BaseMultiPretrained):
    
    """
    Implementation of Mean Teacher as introduced in <https://arxiv.org/abs/1703.01780>

    Args:
        
    alpha (:obj:`float`): memory of the last epochs.
    
    unsup_weight: Initial unsupervised weight.
    
    Class attributes:
         
     -**firstpass**: bool variable to track if its the first pass through the forward method.
     
    """

    def __init__(self, alpha=0.5, unsup_weight=0, *args, **kwargs):
        
        super().__init__(*args, ssl_model_type='MeanTeacher', **kwargs)
        
        if self.teacher_model_name != self.student_model_name:
            warnings.warn("When using different pretrained models for mean teacher, confirm that both have similar parameters, for eg. number and size of hidden layers, attention etc.")
        
        self.apply(self.zero_teacher_weights)
        
        self.unsup_weight=unsup_weight
        self.alpha=alpha
        self.firstpass=True

    def zero_teacher_weights(self, module):
        
        """ 
        Function for zeroing the teachers weights and biases.
        
        """
        
        if hasattr(module, 'weight'):
            module.weight.data.zero_()
        
        if hasattr(module, 'bias'):
            module.bias.data.zero_()
    
        
    def forward(self, **kwargs):
        
        """ 
        Implementation of forward function calculating the semi supervised loss.
        Mixing of the labeled and unlabeled examples in a single batch is not allowed. 
        
        Args:
        
        kwargs: Arguments from pretrained_model.forward.
        
        Return:
        transformers.modeling_outputs.SequenceClassifierOutput object with the 
        information of logits and the loss function.
        
        """

        labels = kwargs.pop("labels")
        stud_logits = self.student(**kwargs).logits
        
        sup_loss = CrossEntropyLoss()
        unsup_loss = MSELoss()
        tot_loss=0
        
        if all(labels>=0):
            tot_loss = sup_loss(stud_logits.view(-1, self.num_labels), labels.view(-1))
        
        teach_logits=None
        
        if self.firstpass==False and self.training:
            
            with torch.no_grad():
                teach_logits = self.teacher(**kwargs).logits
                
            tot_loss += self.unsup_weight*unsup_loss(teach_logits, stud_logits)
        
        elif self.training:
            ##required to free the graph in the firstpass
            tot_loss += 0*stud_logits.sum()

        
        logits = teach_logits if teach_logits is not None else stud_logits
        
        return SequenceClassifierOutput(
            loss=tot_loss,
            logits=logits
        )
    

    def update_teacher_variables(self):

        """
        Function for updating teacher weights and bias. Directly
        used from <https://github.com/CuriousAI/mean-teacher>
                        
        """
        self.firstpass=False

        for teach_param, stud_param in zip(self.teacher.parameters(), self.student.parameters()):
            teach_param.data.mul_(self.alpha).add_(stud_param.data, alpha = 1 - self.alpha)        

        
@add_signature_from(BaseMultiPretrained)                
class NoisyStudent(BaseMultiPretrained):
    
    """
    Implementation of Noisy Student as introduced in <https://arxiv.org/abs/1911.04252>

    Args:
    
    kwargs: keyword arguments are the same as is for BaseMultiPretrained class,
    except model type string. Class forward initialized with teacher as the teacher is trained first. 
    
    """

    def __init__(self, *args, **kwargs):
        
        
        super().__init__(*args, ssl_model_type='NoisyStudent', **kwargs)

        self.teacher_dropout = self.teacher.dropout.p
        self.student_dropout = self.student.dropout.p
        
        if self.teacher_dropout > self.student_dropout:
            warnings.warn("For NoisyStudent to work properly, the teacher classifier dropout should be lower than student classifier dropout.")
        
        self.forward = self.student.forward
