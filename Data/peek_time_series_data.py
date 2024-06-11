import torch
import pickle as pkl
import os


def peek(dataset_dir):

    for file_name in os.listdir(dataset_dir):   # open a specific location
        with open(dataset_dir + file_name, 'rb') as f:
            d = pkl.load(f)

        if len(d['df'].columns) == 46:  # A valid location: Because 23/261 counties has 47 columns, then we set locations having 46 columns as valid locations

            for index, row in d['df'].iterrows():  # for each hour in that specific location, index is the unique timestamp
                # time
                # print(index)

                # weather features
                feature_items = torch.tensor(row.values[:len(row) - 1]).double()
                feature_items = torch.nan_to_num(feature_items, nan=0.0)
                # print(feature_items)

                # extreme weather (i.e., thunderstorm) labels, 0 means not happen
                label_items = torch.unsqueeze(torch.tensor(row.values[len(row) - 1]).double(), 0)
                # print(label_items)

            print("Time-series from " + file_name + " is peeked.")


if __name__ == '__main__':
    input_data_dir = 'Labeled_Time_Series/'
    peek(input_data_dir)

