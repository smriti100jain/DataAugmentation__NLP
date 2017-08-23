import numpy as np


class Gen_Data_loader():
    def __init__(self, batch_size,SEQ_LENGTH):
        self.batch_size = batch_size
        self.token_stream = []
        self.sequence_length = SEQ_LENGTH
    def create_batches(self, data_file):
        self.token_stream = []
        print(data_file)
        with open(data_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                print(len(parse_line))
                if len(parse_line) == self.sequence_length:
                    self.token_stream.append(parse_line)
        self.num_batch = int(len(self.token_stream) / self.batch_size)
        print(self.num_batch)
        print(len(self.token_stream))
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        ret = self.sequence_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0


class Likelihood_data_loader():
    def __init__(self, batch_size,SEQ_LENGTH):
        self.batch_size = batch_size
        self.token_stream = []
        self.sequence_length = SEQ_LENGTH
    def create_batches(self, data_file):
        self.token_stream = []
        with open(data_file, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == self.sequence_length:
                    self.token_stream.append(parse_line)

        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        ret = self.sequence_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0
