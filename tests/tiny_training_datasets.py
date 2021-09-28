from datasets import Dataset, DatasetDict


def correct_examples():

    labeled = Dataset.from_dict(
        {'sentence': ['moon can be red.', 'There are no people on moon.'],
         'label': [1, 0]})

    unlabeled = Dataset.from_dict(
        {'sentence': ['moon what??.', 'I am people'], 'label': [-1, -1]})

    return (labeled, unlabeled)


def get_correct_dataset_TUWS(wrong_key=False):

    labeled, unlabeled = correct_examples()

    train_dic = Dataset.from_dict({
        'sentence': labeled['sentence'] + unlabeled['sentence'],
        'label': labeled['label'] + unlabeled['label']})

    if wrong_key is False:
        return DatasetDict({'train': train_dic})
    else:
        return DatasetDict({'training_Data': train_dic})


def get_wrong_dataset_TUWS():

    labeled, _ = correct_examples()

    unlabeled = Dataset.from_dict(
        {'sentence': ['moon what??.', 'I am people'], 'label': [-1, 0]})

    train_dic = Dataset.from_dict({
        'sentence': labeled['sentence'] + unlabeled['sentence'],
        'label': labeled['label'] + unlabeled['label']})

    return DatasetDict({'train': train_dic})


def get_dataset_cotrain(wrong_key=False):

    labeled, _ = correct_examples()

    unlabeled = Dataset.from_dict(
        {'sentence': ['moon what??.', 'I am people']})

    if wrong_key is False:
        return DatasetDict({'labeled1': labeled,
                            'labeled2': labeled,
                            'unlabeled': unlabeled})
    else:
        return DatasetDict({'labeled1': labeled,
                            'labeled2': labeled,
                            'unlabels': unlabeled})
