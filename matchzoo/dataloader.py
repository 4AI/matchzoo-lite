# -*- coding: utf-8 -*

"""数据加载器件
Description:

1. 如果传入的是文件，要求的格式示例：
   {"text_left": "” 东 坝 是 她 的 一 个 遗 憾", "text_right": "上 面 说 到 , 网 贷 平 台 的 出 现 , 很 好 地 解 决 了 一 些 急 于 用 钱 的 个 人 和 企 业 的 贷 款 难 题", "label": 0}
2. 也可传入 datas 和 labels ，两者都是数组，labels 是 datas 对应的标签

Author: niming.lxm@antfin.com
"""

import os
import typing
import json
import pandas as pd
import matchzoo

def load_data(stage: str = 'train',
              task: str = 'ranking',
              corpus_dir: str = 'corpus',
              datas: list = None,
              labels: list = None) -> typing.Union[matchzoo.DataPack, tuple]:
    """
    Load data.

    :param stage: One of `train`, `dev`, and `test`.
    :param task: Could be one of `ranking`, `classification` or a
        :class:`matchzoo.engine.BaseTask` instance.
    :param corpus_dir: dirname to corpus, which include train.json and test.json
    :param datas: list of tuple like [(sentence1, sentence2), ...]
    :param labels: the labels of datas

    :return: A DataPack if `ranking`, a tuple of (DataPack, classes) if
        `classification`.
    """
    if stage not in ('train', 'dev', 'test', 'predict'):
        raise ValueError(f"{stage} is not a valid stage."
                         f"Must be one of `train`, `dev`, and `test`.")

    if stage == 'predict':
        data_pack = _read_predict(datas, labels)
    else:
        file_path = os.path.join(corpus_dir, f'{stage}.json')
        data_pack = _read_data(file_path)

    if task == 'ranking':
        task = matchzoo.tasks.Ranking()
    if task == 'classification':
        task = matchzoo.tasks.Classification()

    if isinstance(task, matchzoo.tasks.Ranking):
        data_pack.relation['label'] = \
            data_pack.relation['label'].astype('float32')
        return data_pack
    elif isinstance(task, matchzoo.tasks.Classification):
        data_pack.relation['label'] = data_pack.relation['label'].astype(int)
        return data_pack.one_hot_encode_label(num_classes=2), [False, True]
    else:
        raise ValueError(f"{task} is not a valid task.")


def _read_data(fpath):
    datas = []
    with open(fpath, "r") as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            datas.append((obj['label'],
                          " ".join(obj['text_left'].split()),
                          " ".join(obj['text_right'].split())))

    df = pd.DataFrame(data=datas, columns=["label", "text_left", "text_right"])
    return matchzoo.pack(df)


def _read_predict(datas, labels=None):
    if not labels:
        labels = [0] * len(datas)

    dataset = []
    for label, (left, right) in zip(labels, datas):
        dataset.append((label, " ".join(list(left)), " ".join(list(right))))
    df = pd.DataFrame(data=dataset, columns=["label", "text_left", "text_right"])
    return matchzoo.pack(df)

