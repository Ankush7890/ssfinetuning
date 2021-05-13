
Semi Supervised FineTuning of Pretrained Transformer model(NLP) for Sequence Classification
============================================================================================

.. toctree::
   :maxdepth: 2
   :caption: About

This package implements various semisupervised learning (SSL) approaches commonly known in computer vision to NLP, only at the finetuning stage of the models. This repo is created to explore how far can one get by applying ssl at only the classifier layer/layers.

The `sslfinetuning` is implemented using the class composition with Auto classes of `HuggingFace's Transformers <https://github.com/huggingface/transformers>`_ library. So, any pretrained transformer model available at HuggingFace should be able to run here, for Sequence classification.

Here, trainers are deriven from `transformers.Trainer` for different ssl model. 

SSL Models
==========

1. `PiModel <https://arxiv.org/abs/1610.02242>`_ as introduced in paper *Temporal Ensembling for Semi-Supervised Learning* by Samuli Laine, and Timo Aila.

2. `TemporalEnsemble <https://arxiv.org/abs/1610.02242>`_ also introduced in the above paper.

3. `CoTrain <https://www.cs.cmu.edu/~avrim/Papers/cotrain.pdf>`_ as introduced in the paper *Combining Labeled and Unlabeled Data with Co-Training* by Avrim Blum and Tom Mitchell.

4. `TriTrain <https://ieeexplore.ieee.org/document/1512038>`_ was first introduced in the paper *Tri-training: exploiting unlabeled data using three classifiers* by Zhi-Hua Zhou and Ming Li. However, in this project the implementation is more closer to implementation in `Strong Baselines for Neural Semi-supervised Learning under Domain Shift <https://arxiv.org/abs/1804.09530>`_ and `Asymmetric Tri-training for Unsupervised Domain Adaptation <https://arxiv.org/abs/1702.08400>`_. 

5. `MeanTeacher <https://arxiv.org/abs/1703.01780>`_ as introduced in the paper *Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results* by Antti Tarvainen, and Harri Valpola. 

6. `NoisyStudent <https://arxiv.org/abs/1911.04252>`_ as introduced in the paper *Self-training with Noisy Student improves ImageNet classification* by Qizhe Xie, Minh-Thang Luong, Eduard Hovy, and Quoc V. Le.

Getting started
================

1. Important points to consider before using any of the models:

   * To prepare datasets for `PiModel`, `TemporalEnsemble`, and `MeanTeacher`, unlabeled datapoints should be labeled with negative labels ( <=0 ). Also, a batch should not contain mix of labeled and unlabeled datapoints. For eg.
    
    ::
	
        >>> labeled = datasets.Dataset.from_dict({'sentence':['moon can be red.', 'There are no people on moon.'], 'label':[1, 0]}) 

        >>> unlabeled = Dataset.from_dict({'sentence':['moon what??.', 'I am people'], 'label':[-1, -1]}) ##correct way to unlabeled datasets.

        >>> unlabeled_wrong = Dataset.from_dict({'sentence':['moon what??.', 'I am people'], 'label':[0, -1]}) ##wrong way to unlabeled datasets.

        >>> dataset_training = Dataset.from_dict({'sentence':labeled['sentence'] + unlabeled['sentence'], 'label':labeled['label'] + unlabeled['label']})
        


   * If directly using the Trainer from ~trainer_util modules. Following maps between the trainers and models should be considered.
   
   ::

	>>> from trainer_util import (
	>>> TrainerWithUWScheduler, 
	>>> TrainerForCoTraining,
	>>> TrainerForTriTraining,
	>>> TrainerForNoisyStudent)
	
	>>> from models import (
	>>> PiModel,
	>>> TemporalEnsembleModel,
	>>> CoTrain,
	>>> TriTrain,
	>>> MeanTeacher,
	>>> NoisyStudent)

	>>> MAPPING_BETWEEN_TRAINER_AND_MODEL = OrderedDict([
	>>> (PiModel, TrainerWithUWScheduler),
	>>> (TemporalEnsemble, TrainerWithUWScheduler),
	>>> (CoTrain, TrainerForCoTraining),
	>>> (TriTrain, TrainerForTriTraining),
	>>> (MeanTeacher, TrainerWithUWScheduler),
	>>> (NoisyStudent, TrainerForNoisyStudent),
	>>>  ])

        
2. Two ways to train with semi supervised learning.
    
    * Use `training_args.train_with_ssl` which takes care of the above mapping in couple of lines of code.
    
    * Using an appropriate Trainer from 'trainer_util' with the model from `models` as shown in the above mapping.

   
   
Indices and tables
==================

* :ref:`genindex`
