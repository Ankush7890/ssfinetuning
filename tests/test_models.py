import unittest
from ssfinetuning.models import *
from transformers import PreTrainedModel


class TestPiModel(unittest.TestCase):

    def test_for_constructor(self):

        model_pi = PiModel()

        self.assertTrue(hasattr(model_pi, 'pretrained_model'))
        self.assertTrue(isinstance(model_pi.pretrained_model, PreTrainedModel))
        self.assertTrue(hasattr(model_pi, 'unsup_weight'))
        self.assertTrue(hasattr(model_pi, 'simple_forward'))
        self.assertEqual(model_pi.num_models, 1)


class TestTemporalEnsembleModel(unittest.TestCase):

    def test_for_constructor(self):

        model_te = TemporalEnsembleModel()

        self.assertTrue(hasattr(model_te, 'pretrained_model'))
        self.assertTrue(isinstance(model_te.pretrained_model, PreTrainedModel))
        self.assertTrue(hasattr(model_te, 'unsup_weight'))
        self.assertTrue(hasattr(model_te, 'alpha'))
        self.assertTrue(hasattr(model_te, 'logits_batchwise'))
        self.assertTrue(hasattr(model_te, 'simple_forward'))
        self.assertEqual(model_te.num_models, 1)


class TestCoTrain(unittest.TestCase):

    def test_for_constructor(self):

        model_co = CoTrain()

        self.assertTrue(hasattr(model_co, 'pretrained_model'))
        self.assertTrue(isinstance(model_co.pretrained_model, PreTrainedModel))
        self.assertTrue(hasattr(model_co, 'o_weight'))
        self.assertTrue(hasattr(model_co, 'simple_forward'))
        self.assertEqual(model_co.num_models, 2)
        self.assertTrue(hasattr(model_co, 'classifiers'))
        self.assertEqual(len(model_co.classifiers), 2)


class TestTriTrain(unittest.TestCase):

    def test_for_constructor(self):

        model_tri = TriTrain()

        self.assertTrue(hasattr(model_tri, 'pretrained_model'))
        self.assertTrue(
            isinstance(
                model_tri.pretrained_model,
                PreTrainedModel))
        self.assertTrue(hasattr(model_tri, 'o_weight'))
        self.assertTrue(hasattr(model_tri, 'simple_forward'))
        self.assertEqual(model_tri.num_models, 3)
        self.assertTrue(hasattr(model_tri, 'classifiers'))
        self.assertEqual(len(model_tri.classifiers), 3)


class TestMeanTeacher(unittest.TestCase):

    def test_for_constructor(self):

        model_me = MeanTeacher()

        self.assertTrue(hasattr(model_me, 'teacher'))
        self.assertTrue(isinstance(model_me.teacher, PreTrainedModel))

        self.assertTrue(hasattr(model_me, 'student'))
        self.assertTrue(isinstance(model_me.student, PreTrainedModel))

        self.assertTrue(hasattr(model_me, 'unsup_weight'))
        self.assertTrue(hasattr(model_me, 'alpha'))


class TestNoisyStudent(unittest.TestCase):

    def test_for_constructor(self):

        model_ns = NoisyStudent()

        self.assertTrue(hasattr(model_ns, 'teacher'))
        self.assertTrue(isinstance(model_ns.teacher, PreTrainedModel))

        self.assertTrue(hasattr(model_ns, 'student'))
        self.assertTrue(isinstance(model_ns.student, PreTrainedModel))

        self.assertTrue(hasattr(model_ns, 'teacher_dropout'))
        self.assertTrue(hasattr(model_ns, 'student_dropout'))
