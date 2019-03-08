# MatchZoo-Lite

基于 [MatchZoo 2.0.0](https://github.com/NTMC-Community/MatchZoo/tree/v2.0) 开发，并做了简化

## 主要修改

### 增
增加数据加载器 dataloader 可方便进行数据的加载，训练数据和测试数据的文件格式统一为 json 文件，格式为：

```json
{"text_left": "xxx xxx xx", "text_right": "xxx xxx xxx", "label": 1}
```

其中 `text_left` 和 `text_right` 为空格分割的分词文本

### 删改
- 去除 nltk 相关语料库的调用（如停用词）
- 去除预提供的 datasets
- 更换了测试 tests
- 去除部分模型，只保留以下模型：
  - arci
  - arcii
  - dssm
  - cdssm
  - conv_highway
  - duet
  - match_pyramid
  - mvlstm

## Install

MatchZoo is dependent on [Keras](https://github.com/keras-team/keras), please install one of its backend engines: TensorFlow, Theano, or CNTK. We recommend the TensorFlow backend. Two ways to install MatchZoo:

### Install matchzoo-lite from the Github source

```
git clone http://gitlab.alipay-inc.com/niming.lxm/matchzoo-lite.git
cd matchzoo-lite
python setup.py install
```
### Docker

```
docker pull seanlee97/matchzoo-lite:latest
```


## Train your model
- [train_duet.py](http://gitlab.alipay-inc.com/niming.lxm/matchzoo-lite/blob/master/examples/train_duet.py)
- [train_arcii.py](http://gitlab.alipay-inc.com/niming.lxm/matchzoo-lite/blob/master/examples/train_arcii.py)
- [train_dssm.py](http://gitlab.alipay-inc.com/niming.lxm/matchzoo-lite/blob/master/examples/train_dssm.py)
- [train_mvlstm.py](http://gitlab.alipay-inc.com/niming.lxm/matchzoo-lite/blob/master/examples/train_mvlstm.py)

## Get Started in 60 Seconds

To train a [Deep Semantic Structured Model](https://www.microsoft.com/en-us/research/project/dssm/), import matchzoo and prepare input data.

```python
import matchzoo as mz

train_pack = mz.datasets.wiki_qa.load_data('train', task='ranking')
valid_pack = mz.datasets.wiki_qa.load_data('dev', task='ranking')
predict_pack = mz.datasets.wiki_qa.load_data('test', task='ranking')
```

Preprocess your input data in three lines of code, keep track parameters to be passed into the model.

```python
preprocessor = mz.preprocessors.DSSMPreprocessor()
train_pack_processed = preprocessor.fit_transform(train_pack)
valid_pack_processed = preprocessor.transform(valid_pack)
predict_pack_processed = preprocessor.transform(predict_pack)
```

Make use of MatchZoo customized loss functions and evaluation metrics:

```python
ranking_task = mz.tasks.Ranking(loss=mz.losses.RankCrossEntropyLoss(num_neg=4))
ranking_task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
    mz.metrics.MeanAveragePrecision()
]
```

Initialize the model, fine-tune the hyper-parameters.

```python
model = mz.models.DSSM()
model.params['input_shapes'] = preprocessor.context['input_shapes']
model.params['task'] = ranking_task
model.params['mlp_num_layers'] = 3
model.params['mlp_num_units'] = 300
model.params['mlp_num_fan_out'] = 128
model.params['mlp_activation_func'] = 'relu'
model.guess_and_fill_missing_params()
model.build()
model.compile()
```

Generate pair-wise training data on-the-fly, evaluate model performance using customized callbacks on prediction data.

```python
train_generator = mz.PairDataGenerator(train_pack_processed, num_dup=1, num_neg=4, batch_size=64, shuffle=True)

pred_x, pred_y = predict_pack_processed.unpack()
evaluate = mz.callbacks.EvaluateAllMetrics(model, x=pred_x, y=pred_y, batch_size=len(pred_x))

history = model.fit_generator(train_generator, epochs=20, callbacks=[evaluate], workers=5, use_multiprocessing=False)
```

## References
[MatchZoo](https://github.com/NTMC-Community/MatchZoo)

[Tutorials](https://github.com/NTMC-Community/MatchZoo/tree/master/tutorials)

[English Documentation](https://matchzoo.readthedocs.io/en/master/)

[中文文档](https://matchzoo.readthedocs.io/zh/latest/)

If you're interested in the cutting-edge research progress, please take a look at [awaresome neural models for semantic match](https://github.com/NTMC-Community/awaresome-neural-models-for-semantic-match).


## License

[Apache-2.0](https://opensource.org/licenses/Apache-2.0)

## MatchZoo License

[Apache-2.0](https://opensource.org/licenses/Apache-2.0)
Copyright (c) 2015-present, Yixing Fan (faneshion)
