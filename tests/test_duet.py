# -*- coding: utf-8 -*-

import shutil

import numpy as np
import pytest

import matchzoo as mz

corpus_dir = "./corpus"

@pytest.fixture(scope='module')
def train_data():
    return mz.dataloader.load_data(stage='train', task='classification', corpus_dir=corpus_dir)[0]

@pytest.fixture(scope='module')
def test_data():
    return mz.dataloader.load_data(stage='test', task='classification', corpus_dir=corpus_dir)[0]

@pytest.fixture(scope='module')
def task(request) -> mz.engine.BaseTask:
    return mz.tasks.Classification()


@pytest.fixture(scope='module')
def preprocessor():
    return mz.preprocessors.BasicPreprocessor()


@pytest.fixture(scope='module')
def train_data_processed(train_data, preprocessor) -> mz.DataPack:
    data = preprocessor.fit_transform(train_data)
    return data


@pytest.fixture(scope='module')
def test_data_processed(test_data, preprocessor) -> mz.DataPack:
    return preprocessor.transform(test_data)


@pytest.fixture(scope='module')
def train_generator(request, train_data_processed):
    return mz.DataGenerator(train_data_processed)


@pytest.mark.slow
def test_duet(train_data_processed,
                 task,
                 train_generator,
                 test_data_processed,
                 preprocessor):
    """Test DUET model."""
    # Create a duet model
    duet = mz.models.DUET()
    input_shapes = preprocessor.context['input_shapes']
    embed_dimension = preprocessor.context['vocab_size'] + 1
    duet.params['input_shapes'] = input_shapes
    duet.params['task'] = task
    duet.params['embedding_input_dim'] = embed_dimension
    duet.params['embedding_output_dim'] = 10
    duet.params['lm_filters'] =  32
    duet.params['lm_hidden_sizes'] = [16]
    duet.params['dm_filters'] = 32
    duet.params['dm_kernel_size'] = 3
    duet.params['dm_hidden_sizes'] = [16]
    duet.params['dropout_rate'] = 0.5
    duet.params['activation_func'] = 'relu'
    duet.guess_and_fill_missing_params()
    duet.build()
    duet.compile()

    x_test, y_test = test_data_processed.unpack()
    test_eval = mz.callbacks.EvaluateAllMetrics(duet,
                                                 x_test,
                                                 y_test)
    duet.fit_generator(train_generator, epochs=1,
                                callbacks=[test_eval])
    duet.save('.tmpdir')

    try:
        duet = mz.load_model('.tmpdir')
        x, y = test_data_processed.unpack()
        results = duet.evaluate(x, y)
        assert len(results) > 0
    finally:
        shutil.rmtree('.tmpdir')
