import unittest
from ssfinetuning.models import *
from unittest.mock import patch
from torch import tensor
from transformers.modeling_outputs import SequenceClassifierOutput


def get_logits_examples():

    return logits1, logits2, labels_match_1, labels_match_2


class setUpMixin:

    """
    A common setUp method class for TemporalEnsembleModel and PiModel tests.

    """

    def setUp(self):

        # assuming 2 classes
        self.logits1 = SequenceClassifierOutput(logits=tensor([[10.0, 0.0], [10.0, 0.0]]))

        self.logits2 = SequenceClassifierOutput(logits=tensor([[0.0, 10.0], [0.0, 10.0]]))

        # labels that match with logits1
        self.labels_match_1 = {'labels': tensor([0, 0])}

        # labels that match with logits2
        self.labels_match_2 = {'labels': tensor([1, 1])}


class TestPiModel(setUpMixin, unittest.TestCase):

    @patch('ssfinetuning.models.AutoModelForSequenceClassification', autospec=True)
    def test_for_constructor(self, mock_Auto):

        model_pi = PiModel()

        self.assertTrue(hasattr(model_pi, 'pretrained_model'))
        self.assertTrue(mock_Auto.from_pretrained.called)
        self.assertTrue(hasattr(model_pi, 'unsup_weight'))
        self.assertTrue(hasattr(model_pi, 'simple_forward'))
        self.assertEqual(model_pi.num_models, 1)

    @patch('ssfinetuning.models.AutoModelForSequenceClassification')
    def test_forward(self, mock_Auto):

        # setting unsupervised weight to 1
        model_pi = PiModel(unsup_weight=1)
        mock_auto_pr = mock_Auto.from_pretrained()

        mock_auto_pr.forward.side_effect = [self.logits1, self.logits1,
                                            self.logits1, self.logits2,
                                            self.logits2, self.logits2,
                                            self.logits1, self.logits1]

        output1 = model_pi.forward(**self.labels_match_1)

        self.assertLessEqual(abs(output1.loss - 0.0), 1e-4)

        output2 = model_pi.forward(**self.labels_match_1)

        # unsupervised loss ((10-0)^2 + (0-10)^2)/2.0
        self.assertLessEqual(abs(output2.loss - 100.0), 1e-4)

        output3 = model_pi.forward(**self.labels_match_2)

        self.assertLessEqual(abs(output3.loss - 0.0), 1e-4)

        output4 = model_pi.forward(**self.labels_match_2)

        # supervised loss
        #-log(exp(0)) - log(exp(10)) + 2*log(exp(0)+exp(10))
        #              = 10.00
        self.assertLessEqual(abs(output4.loss - 10), 1e-4)


class TestTemporalEnsembleModel(setUpMixin, unittest.TestCase):

    def almostEqualTensors(self, tensor1, tensor2, tolerance=1e-8):

        diff = (abs(tensor1 - tensor2) < tolerance)

        assert diff.all()

    def init_for_memory_logits(self, model):

        # extract logits from SequenceClassifierOutput
        model.logits_batchwise = [self.logits1.logits, self.logits2.logits]

        model.mini_batch_num = 2

        model.update_memory_logits(1)

    @patch('ssfinetuning.models.AutoModelForSequenceClassification', autospec=True)
    def test_for_constructor(self, mock_Auto):

        model_te = TemporalEnsembleModel()

        self.assertTrue(hasattr(model_te, 'pretrained_model'))
        self.assertTrue(mock_Auto.from_pretrained.called)
        self.assertTrue(hasattr(model_te, 'unsup_weight'))
        self.assertTrue(hasattr(model_te, 'alpha'))
        self.assertTrue(hasattr(model_te, 'logits_batchwise'))
        self.assertTrue(hasattr(model_te, 'simple_forward'))
        self.assertEqual(model_te.num_models, 1)

    @patch('ssfinetuning.models.AutoModelForSequenceClassification')
    def test_update_memory_logits(self, mock_Auto):

        model_te = TemporalEnsembleModel(alpha=0)

        self.init_for_memory_logits(model_te)

        self.assertFalse(model_te.firstpass)

        # 0+logit_i=logit_i
        self.almostEqualTensors(model_te.memory_logits[0], self.logits1.logits)

        self.almostEqualTensors(model_te.memory_logits[1], self.logits2.logits)

        model_te.alpha = 0.5

        self.init_for_memory_logits(model_te)

        # (0.5*logit_i + 0.5*logit_i)/0.5 = 2*logit_i
        self.almostEqualTensors(model_te.memory_logits[0], 2 * self.logits1.logits)

        self.almostEqualTensors(model_te.memory_logits[1], 2 * self.logits2.logits)

    @patch('ssfinetuning.models.AutoModelForSequenceClassification')
    def test_forward(self, mock_Auto):

        # setting unsupervised weight to 1
        model_te = TemporalEnsembleModel(alpha=0, unsup_weight=1)
        mock_auto_pr = mock_Auto.from_pretrained()

        mock_auto_pr.forward.side_effect = [self.logits1,
                                            self.logits1,
                                            self.logits2,
                                            self.logits1]

        output1 = model_te.forward(**self.labels_match_1)

        self.assertLessEqual(abs(output1.loss - 0.0), 1e-4)

        output2 = model_te.forward(**self.labels_match_2)

        # supervised loss: see Model Pi test.
        self.assertLessEqual(abs(output2.loss - 10), 1e-4)

        model_te.update_memory_logits(1)

        output3 = model_te.forward(**self.labels_match_2)

        # unsupervised loss ((10-0)^2 + (0-10)^2)/2.0
        self.assertLessEqual(abs(output3.loss - 100), 1e-4)

        output4 = model_te.forward(**self.labels_match_1)

        # since memory is set to 100% as alpha=1
        self.assertLessEqual(abs(output4.loss - 0), 1e-4)


class TestCoTrain(unittest.TestCase):

    @patch('ssfinetuning.models.AutoModelForSequenceClassification', autospec=True)
    def test_for_constructor(self, mock_Auto):

        model_co = CoTrain()

        self.assertTrue(hasattr(model_co, 'pretrained_model'))
        self.assertTrue(mock_Auto.from_pretrained.called)
        self.assertTrue(hasattr(model_co, 'o_weight'))
        self.assertTrue(hasattr(model_co, 'simple_forward'))
        self.assertEqual(model_co.num_models, 2)
        self.assertTrue(hasattr(model_co, 'classifiers'))
        self.assertEqual(len(model_co.classifiers), 2)


class TestTriTrain(unittest.TestCase):

    @patch('ssfinetuning.models.AutoModelForSequenceClassification', autospec=True)
    def test_for_constructor(self, mock_Auto):

        model_tri = TriTrain()

        self.assertTrue(hasattr(model_tri, 'pretrained_model'))
        self.assertTrue(mock_Auto.from_pretrained.called)
        self.assertTrue(hasattr(model_tri, 'o_weight'))
        self.assertTrue(hasattr(model_tri, 'simple_forward'))
        self.assertEqual(model_tri.num_models, 3)
        self.assertTrue(hasattr(model_tri, 'classifiers'))
        self.assertEqual(len(model_tri.classifiers), 3)


class TestMeanTeacher(unittest.TestCase):

    @patch('ssfinetuning.models.AutoModelForSequenceClassification', autospec=True)
    def test_for_constructor(self, mock_Auto):

        model_me = MeanTeacher()

        self.assertTrue(hasattr(model_me, 'teacher'))
        self.assertTrue(hasattr(model_me, 'student'))

        self.assertTrue(mock_Auto.from_pretrained.call_count, 2)

        self.assertTrue(hasattr(model_me, 'unsup_weight'))
        self.assertTrue(hasattr(model_me, 'alpha'))


class TestNoisyStudent(unittest.TestCase):

    @patch('ssfinetuning.models.AutoModelForSequenceClassification')
    def test_for_constructor(self, mock_Auto):

        mock_auto_pr = mock_Auto.from_pretrained()
        mock_auto_pr.dropout.p = 0

        model_ns = NoisyStudent()

        self.assertTrue(hasattr(model_ns, 'teacher'))
        self.assertTrue(hasattr(model_ns, 'student'))

        self.assertTrue(mock_Auto.from_pretrained.call_count, 2)

        self.assertTrue(hasattr(model_ns, 'teacher_dropout'))
        self.assertTrue(hasattr(model_ns, 'student_dropout'))
