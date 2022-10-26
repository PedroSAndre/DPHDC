from torchhd.datasets import ISOLET


def readDataset(train: bool) -> tuple[list[list[int]], list[str]]:
    data_wrapper = ISOLET("dataset/", train=train, download=True)

    data_int = []
    for i in data_wrapper.data.tolist():
        data_int.append([])
        for j in i:
            aux: int = int((j + 1) * 10)
            if aux != 20:
                data_int[-1].append(aux)
            else:
                data_int[-1].append(aux - 1)

    data_labels = []
    for i in data_wrapper.targets.tolist():
        data_labels.append(data_wrapper.classes[i])

    return data_int, data_labels
