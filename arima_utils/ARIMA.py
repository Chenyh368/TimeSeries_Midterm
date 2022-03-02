from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from dl_utils.metrics import *

class ExpArima(object):
    def __init__(self, args):
        print('====> Build ARIMA Model')
        '''
        :param d: 差分次数
        :param p: 自回归阶数
        :param q: 移动平均阶数
        '''
        #================ ARIMA ===============
        self.d = args.d
        self.p = args.p
        self.q = args.q
        # ================ DATA ===============
        self.root_path = args.root_path
        self.data_path = args.data_path
        self.target = args.target
        self.split = args.split
        if args.scaler:
            self.scaler = StandardScaler()
        else:
            self.scaler = None

    def _get_data(self, flag='train'):
        assert flag in ['train', 'test', 'all']
        type_map = {'train': 0, 'test': 1, 'all': 2}
        set_type = type_map[flag]

        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        num_train = int(len(df_raw) * self.split)
        num_test = len(df_raw) - num_train
        border1s = [0, num_train, 0]
        border2s = [num_train, len(df_raw), len(df_raw)]
        border1 = border1s[set_type]
        border2 = border2s[set_type]

        # get target data
        df_data = df_raw[[self.target]][border1:border2]

        if self.scaler:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # time stamp
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        self.data = data
        self.date = df_stamp

    def train(self):
        pass

    def test(self):
        pass

    def get_data(self, flag='train'):
        self._get_data(flag)
        return self.date, self.data

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

