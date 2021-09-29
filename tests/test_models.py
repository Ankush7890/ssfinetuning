import unittest
from ssfinetuning.models import *
from unittest.mock import patch
from torch import tensor
from transformers.modeling_outputs import SequenceClassifierOutput


class TestPiModel(unittest.TestCase):

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

        # assuming 2 classes
        logits1 = SequenceClassifierOutput(
            logits=tensor([[10.0, 0.0], [10.0, 0.0]]))
        logits2 = SequenceClassifierOutput(
            logits=tensor([[0.0, 10.0], [0.0, 10.0]]))

        mock_auto_pr.forward.side_effect = [logits1, logits1,
                                            logits1, logits2,
                                            logits2, logits2]
        # labels that match with logits1
        labels_match_1 = {'labels': tensor([0, 0])}

        # labels that match with logits2
        labels_match_2 = {'labels': tensor([1, 1])}

        output1 = model_pi.forward(**labels_match_1)

        self.assertLessEqual(abs(output1.loss - 0.0), 1e-4)

        output2 = model_pi.forward(**labels_match_1)

        # unsupervised loss ((10-0)^2 + (0-10)^2)/2.0
        self.assertLessEqual(abs(output2.loss - 100.0), 1e-4)

        output3 = model_pi.forward(**labels_match_2)

        self.assertLessEqual(abs(output3.loss - 0.0), 1e-4)


class TestTemporalEnsembleModel(unittest.TestCase):

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
