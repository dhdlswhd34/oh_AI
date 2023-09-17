import torch
import pickle
import random
import pandas as pd

from model import PaddingLstmodel
from model_util import make_padded_tensor

# 하이퍼파라메터
INPUT_SIZE = 6
HIDDEN_SIZE = 128
LAYER_SIZE = 2
OUTPUT_SIZE = 2
BATCH_SIZE = 16


"""
    데이터 -> 리스트
    현재 데이터 요소: ['vp', 'blood flow', 'presbp', 'predbp', 'prepls', 'age']
    -> 변경 가능
    shape : (데이터 개수, 데이터 stack, 데이터 요소)
    example(
        [
            [
                [70, 250, 140, 80, 76, 67],
                [65, 255, 150, 80, 72, 67],
                [60, 250, 170, 90, 76, 67]
            ],
            [
                [100, 260, 150, 80, 72, 67],
                [60, 265, 160, 90, 76, 67],
                [60, 260, 160, 90, 80, 67],
                [70, 250, 140, 80, 76, 67]
            ],
            [
                [100, 260, 150, 80, 72, 67],
                [60, 265, 160, 90, 76, 67],
                [60, 260, 160, 90, 80, 67]
            ],
            [
                [100, 260, 150, 80, 72, 67],
                [60, 265, 160, 90, 76, 67],
                [60, 260, 160, 90, 80, 67],
                [70, 250, 140, 80, 76, 67],
                [65, 255, 150, 80, 72, 67]
            ],
        ]
    ) -> shape: (4, 2~5, 6)
"""


# 모델 예측
def model_run(model, data):
    # 데이터 padding
    padded_data, data_length = make_padded_tensor(data)

    # 예측
    out = model(padded_data, data_length)
    predictions = out.max(dim=1)[1]

    return predictions


# 테스트 데이터 샘플링
def sampling(data):
    idx = random.sample(range(0, len(data)), 10)
    temp = []
    for c in idx:
        temp.append(data[c])
    return temp, idx


if __name__ == "__main__":

    path = 'data'

    # 데이터 load
    with open(f'{path}\\rnn_zero_data.pkl', "rb") as f:
        zero_data = pickle.load(f)

    with open(f'{path}\\rnn_one_data.pkl', "rb") as f:
        one_data = pickle.load(f)

    # 테스트용 데이터 샘플링
    data, idx = sampling(zero_data)

    model_path = "model\\model.pt"

    # model 선언
    model = PaddingLstmodel(
                        INPUT_SIZE,
                        HIDDEN_SIZE,
                        LAYER_SIZE,
                        OUTPUT_SIZE,
                        bidirectional=True
                        )

    # 가중치 적용
    model.load_state_dict(torch.load(model_path))

    # 평가 모드
    model.eval()

    # 예측
    predict = model_run(model, data)

    df = pd.DataFrame(
        {
            "index": idx,
            "predict": predict,
        }
    )
    print(df)
