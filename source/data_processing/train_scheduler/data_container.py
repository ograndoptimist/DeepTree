from collections import deque


class DataContainer(object):
    def __init__(self):
        self.data = deque()

    def add(self, metadata):
        data = DataContainer.__transform_data(metadata)
        self.data.append(data)
        return data

    @staticmethod
    def __transform_data(metadata):
        return {'metadata': metadata}

    def is_empty(self):
        return len(self) == 0

    def popleft(self):
        return self.data.popleft()

    def appendright(self, item):
        return self.add(item)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return self.data.__repr__()
