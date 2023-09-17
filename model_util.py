import torch
from torch.nn.utils.rnn import pad_sequence


# padded 데이터 생성
def make_padded_tensor(tensor):
    length = []
    padded_tensor = []
    for t in tensor:
        length.append(len(t))
        padded_tensor.append(torch.Tensor(t))
    padded_tensor = pad_sequence(padded_tensor, batch_first=True)
    return padded_tensor, length
