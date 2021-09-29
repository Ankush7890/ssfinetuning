import unittest
import ssfinetuning.models
from ssfinetuning.models import *
import transformers.trainer
from ssfinetuning.trainer_util import *
from datasets import Dataset, DatasetDict
from ssfinetuning.default_args import encode
from transformers import TrainingArguments
from unittest.mock import patch, Mock
from torch.optim.lr_scheduler import MultiplicativeLR
import torch

from .tiny_training_datasets import (
    get_correct_dataset_TUWS,
    get_wrong_dataset_TUWS,
    get_dataset_cotrain)
import warnings

warnings.filterwarnings('ignore')


def setup_default_config(dataset):

    args_ta = TrainingArguments(**{'output_dir': "glue",
                                   'evaluation_strategy': 'no',
                                   'learning_rate': 2e-5,
                                   'per_device_train_batch_size': 2,
                                   'num_train_epochs': 4,
                                   'save_steps': 10,
                                   'disable_tqdm': True,
                                   'no_cuda': True})

    encoded, tokenizer = encode(dataset)

    return encoded, tokenizer, args_ta


class TestTrainerWithUWScheduler(unittest.TestCase):

    def setUp(self):

        self.wrong_dataset = get_wrong_dataset_TUWS()

        self.correct_dataset = get_correct_dataset_TUWS()

        self.correct_dataset_wrong_keys = get_correct_dataset_TUWS(
            wrong_key=True)

        self.encoded_dataset, self.tokenizer, self.args_ta = setup_default_config(
            self.correct_dataset)

    def test_for_constructor_errors(self):

        mock_pi = Mock()
        with self.assertRaises(RuntimeError):
            TrainerWithUWScheduler(
                model=mock_pi,
                args=self.args_ta,
                dataset=self.wrong_dataset)

        with self.assertRaises(KeyError):
            TrainerWithUWScheduler(
                model=mock_pi,
                args=self.args_ta,
                dataset=self.correct_dataset_wrong_keys)

    @patch.object(ssfinetuning.trainer_util.Trainer, 'train')
    @patch('ssfinetuning.models.PiModel', autospec=True)
    def test_with_pi(self, mock_pi, mock_trainer):

        mock_pi_obj = mock_pi.to()
        mock_pi_obj.unsup_weight = 0

        trainer = TrainerWithUWScheduler(
            model=mock_pi_obj,
            dataset=self.encoded_dataset,
            args=self.args_ta,
            tokenizer=self.tokenizer,
            unsup_start_epochs=1,
            max_w=1,
            w_ramprate=1,
            update_weights_steps=1
        )

        trainer.train()

        mock_trainer.assert_called()

        # assuming 1 step per epoch
        trainer.create_optimizer_and_scheduler(num_training_steps=4)

        trainer.state.epoch = 0
        trainer.state.global_step = 0

        self.assertTrue(isinstance(trainer.lr_scheduler, UWScheduler))

        # calling the optimizer and lr scheduler
        trainer.optimizer.step()
        trainer.lr_scheduler.step()

        self.assertEqual(mock_pi_obj.unsup_weight, 0)

        # step1=epoch1
        trainer.state.epoch = 1
        trainer.state.global_step = 1

        trainer.lr_scheduler.step()

        self.assertEqual(mock_pi_obj.unsup_weight, 0)

        # step2=epoch2
        trainer.state.epoch = 2
        trainer.state.global_step = 2

        trainer.lr_scheduler.step()

        self.assertEqual(mock_pi_obj.unsup_weight, 1)

        # step3=epoch3
        trainer.state.epoch = 3
        trainer.state.global_step = 3

        trainer.lr_scheduler.step()
        # no change expected since maximum weight(max_w) is 1
        self.assertEqual(mock_pi_obj.unsup_weight, 1)

    @patch('ssfinetuning.models.TemporalEnsembleModel', autospec=True)
    def test_with_te(self, mock_te):

        mock_te_obj = mock_te.to()
        mock_te_obj.type_ = 'TemporalEnsembleModel'

        trainer = TrainerWithUWScheduler(
            model=mock_te_obj,
            dataset=self.encoded_dataset,
            args=self.args_ta,
            tokenizer=self.tokenizer,
            unsup_start_epochs=1,
            max_w=1,
            w_ramprate=1,
            update_weights_steps=1
        )

        # assuming 2 step per epoch
        trainer.create_optimizer_and_scheduler(num_training_steps=8)

        trainer.state.epoch = 0
        trainer.state.global_step = 0

        self.assertTrue(isinstance(trainer.lr_scheduler, UWScheduler))

        # calling the optimizer and lr scheduler
        trainer.optimizer.step()
        trainer.lr_scheduler.step()

        self.assertFalse(mock_te_obj.update_memory_logits.called)

        # step1=epoch0
        trainer.state.epoch = 0
        trainer.state.global_step = 1

        trainer.lr_scheduler.step()

        mock_te_obj.update_memory_logits.assert_called()

        # step2=epoch1
        trainer.state.epoch = 1
        trainer.state.global_step = 2

        trainer.lr_scheduler.step()
        # only called after an epoch is completed
        self.assertEqual(mock_te_obj.update_memory_logits.call_count, 1)

        # step3=epoch1
        trainer.state.epoch = 1
        trainer.state.global_step = 3

        trainer.lr_scheduler.step()

        self.assertEqual(mock_te_obj.update_memory_logits.call_count, 2)

    @patch('ssfinetuning.models.MeanTeacher', autospec=True)
    def test_with_me(self, mock_me):

        mock_me_obj = mock_me.to()
        mock_me_obj.type_ = 'MeanTeacher'

        trainer = TrainerWithUWScheduler(
            model=mock_me_obj,
            dataset=self.encoded_dataset,
            args=self.args_ta,
            tokenizer=self.tokenizer,
            update_teacher_steps=1
        )

        # assuming 1 step per epoch
        trainer.create_optimizer_and_scheduler(num_training_steps=1)

        trainer.state.epoch = 0
        trainer.state.global_step = 0

        self.assertTrue(isinstance(trainer.lr_scheduler, UWScheduler))

        # calling the optimizer and lr scheduler
        trainer.optimizer.step()
        trainer.lr_scheduler.step()

        self.assertFalse(mock_me_obj.update_teacher_variables.called)

        # step1=epoch1
        trainer.state.epoch = 1
        trainer.state.global_step = 1

        trainer.lr_scheduler.step()

        mock_me_obj.update_teacher_variables.assert_called()

        # step2=epoch2
        trainer.state.epoch = 2
        trainer.state.global_step = 2

        trainer.lr_scheduler.step()

        self.assertEqual(mock_me_obj.update_teacher_variables.call_count, 2)


class TestTrainerCoTrain(unittest.TestCase):

    def setUp(self):

        self.dataset = get_dataset_cotrain()

        self.dataset_wrong_key = get_dataset_cotrain(wrong_key=True)

        self.encoded_dataset, self.tokenizer, self.args_ta = setup_default_config(
            self.dataset)

        def dummy_forward(attention_mask, input_ids, label): pass
        # signature is required for mock model objects.
        self.sig = inspect.signature(dummy_forward)

    def test_for_constructor(self):

        mock_co = Mock()

        trainer = TrainerForCoTraining(
            model=mock_co,
            args=self.args_ta,
            dataset=self.dataset)

        with self.assertRaises(KeyError):
            TrainerForCoTraining(
                model=mock_co,
                args=self.args_ta,
                dataset=self.dataset_wrong_key)

    @patch.object(ssfinetuning.trainer_util.TrainerForCoTraining,'exchange_unlabeled_data')
    @patch.object(ssfinetuning.trainer_util.TrainerForCoTraining, 'cotrain')
    @patch('ssfinetuning.models.CoTrain', autospec=True)
    def test_for_train(self, mock_CT, mock_cotrain, mock_ex_unl):

        # setting max_passes to 0 and
        # use_min_lr_scheduler to True
        trainer1 = TrainerForCoTraining(
            model=mock_CT.to(),
            dataset=self.encoded_dataset,
            args=self.args_ta,
            tokenizer=self.tokenizer,
            use_min_lr_scheduler=True,
            max_passes=0
        )

        trainer1.train()

        mock_cotrain.assert_called()

        self.assertTrue(isinstance(trainer1.lr_scheduler, MultiplicativeLR))

        # setting max_passes to 1
        # use_min_lr_scheduler to None
        trainer2 = TrainerForCoTraining(
            model=mock_CT.to(),
            dataset=self.encoded_dataset,
            args=self.args_ta,
            tokenizer=self.tokenizer,
            use_min_lr_scheduler=None,
            max_passes=1
        )

        mock_ex_unl.return_value = False

        trainer2.train()

        self.assertTrue(mock_cotrain.call_count, 3)

        mock_ex_unl.assert_called()

        self.assertFalse(isinstance(trainer2.lr_scheduler, MultiplicativeLR))

    @patch('ssfinetuning.models.CoTrain', autospec=True)
    def test_for_cotrain(self, mock_CT):

        mock_CT_obj = mock_CT.to()

        mock_CT_obj.pretrained_model.forward.__signature__ = self.sig

        # setting epoch_per_cotrain to 2
        trainer = TrainerForCoTraining(
            model=mock_CT_obj,
            dataset=self.encoded_dataset,
            args=self.args_ta,
            tokenizer=self.tokenizer,
            epoch_per_cotrain=2
        )

        trainer.pre_train_init(num_training_steps=4)

        trainer.cotrain(trainer.dataset_model1, trainer.dataset_model2)

        self.assertTrue(mock_CT_obj.cotrain_forward.call_count, 4)

        with patch.object(TrainerForCoTraining, 'get_dataloader') as mock_dl:
            trainer.cotrain(trainer.dataset_model1, trainer.dataset_model2)

        self.assertTrue(mock_dl.call_count, 2)

        with patch.object(TrainerForCoTraining, 'equate_lengths') as mock_eql:
            trainer.cotrain(trainer.dataset_model1, trainer.dataset_model2)

        mock_eql.assert_called()

        # setting epoch_per_cotrain to 4
        trainer2 = TrainerForCoTraining(
            model=mock_CT_obj,
            dataset=self.encoded_dataset,
            args=self.args_ta,
            tokenizer=self.tokenizer,
            epoch_per_cotrain=4
        )

        trainer2.pre_train_init(num_training_steps=8)

        trainer2.cotrain(trainer2.dataset_model1, trainer2.dataset_model2)

        self.assertTrue(mock_CT_obj.cotrain_forward.call_count, 8 + 4)

    @patch('ssfinetuning.models.CoTrain', autospec=True)
    def test_for_exchange_unlabeled_data(self, mock_CT):

        mock_CT_obj = mock_CT.to()

        mock_CT_obj.pretrained_model.forward.__signature__ = self.sig

        # setting p_threshold to 0.65 and
        # exchange_threshold=0
        trainer = TrainerForCoTraining(
            model=mock_CT_obj,
            dataset=self.encoded_dataset,
            args=self.args_ta,
            tokenizer=self.tokenizer,
            p_threshold=0.65,
            exchange_threshold=0
        )

        trainer.pre_train_init(num_training_steps=8)
        # assuming 2 classes
        logits_1 = torch.tensor([[0.70, 0.30], [0.5, 0.5]])
        logits_2 = torch.tensor([[0.40, 0.60], [0.5, 0.5]])

        # simulating the exchange using mock on CoTrain.simple_forward()
        # first step is exchange, second step is not as logits are same in step 1
        # and so on.
        mock_CT_obj.simple_forward.side_effect = [logits_1, logits_2,
                                                  logits_1, logits_2,
                                                  logits_2, logits_1]

        self.assertTrue(trainer.exchange_unlabeled_data())

        self.assertFalse(trainer.exchange_unlabeled_data())

        self.assertTrue(trainer.exchange_unlabeled_data())
