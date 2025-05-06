import torch
import numpy as np
import scipy.io as spio
import h5py

def load_data(data_dir, batch_size, n_data, train_or_test):
    """Return data loader

    Args:
        data_dir: directory to .mat file
        batch_size (int): mini-batch size for loading data

    Returns:
        (data_loader (torch.utils.data.DataLoader), stats)
    """
    
    f = h5py.File(data_dir,'r')
    #f = spio.loadmat(data_dir)
    allDataSize = np.shape(f['dataIn'])[-1]
    x_data = np.transpose(f['dataIn'][:,:,:,:].astype(np.float32))
    #x_data = x_dataAll[:n_data]

    y_data = np.transpose(f['dataOut'][:,:,:,:].astype(np.float32))
    #y_data = y_dataAll[:ndata]
    print("input data shape: {}".format(x_data.shape))
    print("output data shape: {}".format(y_data.shape))

    kwargs = {'num_workers': 4,
              'pin_memory': True} if torch.cuda.is_available() else {}

    dataset = TensorDataset(torch.tensor(x_data), torch.tensor(y_data))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=not(train_or_test), **kwargs)

    fieldMatAll = torch.tensor(np.array(f['fieldMatAll']).transpose())

    return data_loader, fieldMatAll

def make_larger_sensors(sensor_map):
    max_x = sensor_map.size(dim=1)
    max_y = sensor_map.size(dim=0)
    out_sensor_map = torch.zeros(max_y, max_x)
    for y in range(sensor_map.size(dim=0)):
        for x in range(sensor_map.size(dim=1)):
            if sensor_map[y, x]:
                out_sensor_map[y, x] = 1

                back_y = y - 1 >= 0
                for_y = y + 1 < max_y

                back_x = x - 1 >= 0
                for_x = x + 1 < max_x

                if back_y:
                    out_sensor_map[y - 1, x] = 1

                if back_x:
                    out_sensor_map[y, x - 1] = 1

                if for_y:
                    out_sensor_map[y + 1, x] = 1

                if for_x:
                    out_sensor_map[y, x + 1] = 1
    return out_sensor_map