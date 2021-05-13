import unittest
from ssfinetuning.models import *
from ssfinetuning.trainer_util import *
from datasets import Dataset, DatasetDict
from ssfinetuning.default_args import encode
from transformers import TrainingArguments
from unittest.mock import patch, Mock
import warnings
warnings.filterwarnings('ignore')

labeled = Dataset.from_dict({'sentence':['moon can be red.', 'There are no people on moon.'], 'label':[1, 0]})
unlabeled = Dataset.from_dict({'sentence':['moon what??.', 'I am people'], 'label':[-1, -1]})



wrong_unlabeled = Dataset.from_dict({'sentence':['moon what??.', 'I am people'], 'label':[-1, 0]})
train_correct = Dataset.from_dict({'sentence':labeled['sentence'] + unlabeled['sentence'], 'label':labeled['label'] + unlabeled['label']})
train_wrong = Dataset.from_dict({'sentence':labeled['sentence'] + wrong_unlabeled['sentence'], 'label':labeled['label'] + wrong_unlabeled['label']})

correct_ds_for_pi_te_mean = DatasetDict({
    'train' : train_correct,
    } )

wrong_key_ds_for_pi_te_mean = DatasetDict({
    'training_Data' : train_correct,
    } )

wrong_ds_for_pi_te_mean = DatasetDict({
    'train' : train_wrong,
    } )

args_ta = TrainingArguments(**{'output_dir':"glue",
                  'evaluation_strategy':'no',
                  'learning_rate':2e-5,
                  'per_device_train_batch_size':2,
                  'num_train_epochs':2,
                  'save_steps':10,
                  'disable_tqdm':True,
                  'no_cuda' :True})
encoded_pi, tokenizer_pi = encode(correct_ds_for_pi_te_mean) 

class TestTrainerWithUWScheduler(unittest.TestCase):

    def test_for_constructor(self):
        
        
        trainer_pi = TrainerWithUWScheduler(model=PiModel(), dataset = correct_ds_for_pi_te_mean)

        with self.assertRaises(KeyError): 
            TrainerWithUWScheduler(model=PiModel(), dataset = wrong_key_ds_for_pi_te_mean)
        
        trainer_te = TrainerWithUWScheduler(model=TemporalEnsembleModel(), dataset = correct_ds_for_pi_te_mean)
        
        with self.assertRaises(KeyError):
            TrainerWithUWScheduler(model=TemporalEnsembleModel(), dataset = wrong_key_ds_for_pi_te_mean)
        
        trainer_me = TrainerWithUWScheduler(model=MeanTeacher(), dataset = correct_ds_for_pi_te_mean)

        with self.assertRaises(KeyError):
            TrainerWithUWScheduler(model=MeanTeacher(), dataset = wrong_key_ds_for_pi_te_mean)
        
        with self.assertRaises(RuntimeError):
            TrainerWithUWScheduler(model=PiModel(), args=args_ta, dataset=wrong_ds_for_pi_te_mean)
    
    def test_for_train(self):
        
        trainer_pi_1 = TrainerWithUWScheduler(model = PiModel(), args=args_ta, tokenizer=tokenizer_pi, dataset = correct_ds_for_pi_te_mean)
                
        with self.assertRaises(AssertionError):
            trainer_pi_1.train()
        
        #trainer_pi_2 = TrainerWithUWScheduler(model = PiModel(), args=args_ta, tokenizer=tokenizer_pi,  dataset = encoded_pi)
        
        #trainer_pi_2.train()

        trainer_te_1 = TrainerWithUWScheduler(model = TemporalEnsembleModel(), args=args_ta, tokenizer=tokenizer_pi ,dataset = correct_ds_for_pi_te_mean)
        
        with self.assertRaises(AssertionError):
            trainer_te_1.train()

        #trainer_te_2 = TrainerWithUWScheduler(model = TemporalEnsembleModel(), args=args_ta, tokenizer=tokenizer_pi, dataset = encoded_pi)
        
        #trainer_te_2.train()

  
        trainer_me_1 = TrainerWithUWScheduler(model = MeanTeacher(), args=args_ta, tokenizer=tokenizer_pi ,dataset = correct_ds_for_pi_te_mean)
        
        with self.assertRaises(AssertionError):
            trainer_me_1.train()

        #trainer_me_2 = TrainerWithUWScheduler(model = MeanTeacher(), args=args_ta, tokenizer=tokenizer_pi, dataset = encoded_pi)
        
        #trainer_me_2.train()

unlabeled_co = Dataset.from_dict({'sentence':['moon what??.', 'I am people']})

correct_ds_for_co = DatasetDict({
    'labeled1' : labeled,
    'labeled2' : labeled,
    'unlabeled': unlabeled_co
    
    } )

wrong_key_ds_for_co = DatasetDict({
    'labels1' : labeled,
    'labels2' : labeled,
    'unlabels': unlabeled_co
    } )

encoded_co, tokenizer_co = encode(correct_ds_for_co) 

class TestTrainerForCoTraining(unittest.TestCase):
    

    def test_for_constructor(self):
        
        trainer_1 = TrainerForCoTraining(model=CoTrain(), dataset = correct_ds_for_co)

        with self.assertRaises(KeyError): 
            TrainerForCoTraining(model=CoTrain(), dataset = wrong_key_ds_for_co)
    def test_for_train(self):

        trainer = TrainerForCoTraining(model=CoTrain(),args=args_ta, tokenizer=tokenizer_co, dataset = encoded_co)
        
        with patch.object(TrainerForCoTraining, 'cotrain', return_value=None) as mock_cotrain:
            trainer.train()
            
        mock_cotrain.assert_called()
        
        with patch.object(TrainerForCoTraining, 'exchange_unlabeled_data', return_value=bool) as mock_exchange_ud:
            trainer.train()
            
        mock_exchange_ud.assert_called()

correct_ds_for_tri = DatasetDict({
    'labeled1' : labeled,
    'labeled2' : labeled,
    'labeled3' : labeled,
    'unlabeled': unlabeled_co
    
    } )

wrong_key_ds_for_tri = DatasetDict({
    'labels1' : labeled,
    'labels2' : labeled,
    'labels3' : labeled,
    'unlabels': unlabeled_co
    } )

encoded_tri, tokenizer_tri = encode(correct_ds_for_tri) 

class TestTrainerForTriTraining(unittest.TestCase):
    

    def test_for_constructor(self):
        
        trainer_1 = TrainerForTriTraining(model=TriTrain(), dataset = correct_ds_for_tri)

        with self.assertRaises(KeyError): 
            TrainerForTriTraining(model=TriTrain(), dataset = wrong_key_ds_for_tri)

    def test_for_train(self):
                
        trainer = TrainerForTriTraining(model=TriTrain(), args=args_ta, tokenizer=tokenizer_tri, dataset = encoded_tri)
        
        with patch.object(TrainerForTriTraining, 'tri_train', return_value=None) as mock_tritrain:
            trainer.train()
            
        mock_tritrain.assert_called()
        
        with patch.object(TrainerForTriTraining, 'exchange_unlabeled_data', return_value=bool) as mock_exchange_ud:
            trainer.train()
            
        mock_exchange_ud.assert_called()
        

correct_ds_for_ns = DatasetDict({
    'labeled' : labeled,
    'unlabeled': unlabeled_co
    
    } )

wrong_key_ds_for_ns = DatasetDict({
    'label1' : labeled,
    'unlabels': unlabeled_co
    } )

encoded_ns, tokenizer_ns = encode(correct_ds_for_ns) 

class TestTrainerForNoisyStudent(unittest.TestCase):
    
    def test_for_constructor(self):
        
        trainer_1 = TrainerForNoisyStudent(model=NoisyStudent(),  dataset = correct_ds_for_ns)

        with self.assertRaises(KeyError): 
            TrainerForNoisyStudent(model=NoisyStudent(), dataset = wrong_key_ds_for_ns)

    def test_for_train(self):
                
        trainer = TrainerForNoisyStudent(model=NoisyStudent(), args=args_ta, tokenizer=tokenizer_ns, dataset = encoded_ns)
        
        with patch.object(TrainerForNoisyStudent, 'train_and_reset', return_value=None) as mock_train_and_reset:
            trainer.train()
            
        mock_train_and_reset.assert_called()
        
        with patch.object(TrainerForNoisyStudent, 'exchange_models', return_value=bool) as mock_exchange_md:
            trainer.train()
            
        mock_exchange_md.assert_called()
