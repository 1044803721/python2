"""
    模型训练、预测完成后  可以统计分类结果的 混淆矩阵、分类报告
    混淆矩阵 : 矩阵  行：分类 列：分类
    分类报告 ：召回率 F1分数 等结果
"""
import numpy as np
import sklearn.naive_bayes as nb
import matplotlib.pyplot as mp
import sklearn.model_selection as ms
import sklearn.metrics as sm
import yaml
from keras.models import load_model
import matplotlib.pyplot as plt
# 加载数据
from utils import Trainer, Parser
from visual import plot_confusion_matrix

'''data = np.loadtxt("D:\\pythonProject\\MSRLSTM-open\\public\lhy\\data\data_classed_by_label_integrated\\Label_9.txt", delimiter=",")
x = data[:, :9].astype("float")
y = data[:, -1].astype("float")
print(x.shape, x.dtype)
print(y.shape, y.dtype)

train_x, test_x, train_y, test_y = \
    ms.train_test_split(x, y, test_size=0.5, random_state=2)'''
test_data = np.load("D:\\pythonProject\\MSRLSTM-open\\all_data_test_0.5_window_300_overlap_0_no_smooth.npz")
test_x = test_data["x"]
test_y = test_data["y"]
parser = Parser()
parser.create_parser()
pargs = parser.parser.parse_args()
if pargs.config is not None:
    with open(pargs.config, 'r') as f:
        default_arg = yaml.load(f, Loader=yaml.FullLoader)
    key = vars(pargs).keys()
    for k in default_arg.keys():
        if k not in key:
            print('WRONG ARG: {}'.format(k))
            assert (k in key)
    parser.parser.set_defaults(**default_arg)
args = parser.parser.parse_args()
print("参数:", pargs)
mode = args.mode
trainer = Trainer(args.train_args)
batch_size, data_type, device_id, epoch, model_name, train_path, validate_path, window_size = trainer._read_args(trainer.train_args)
trainer._initial_gpu_env(device_id)
trainer._choose_dataset(data_type, test_x, test_x)
print(test_x.shape)
model = load_model("D:/pythonProject/MSRLSTM-open/attention_gg.h5")
print(model.summary())
# model.fit(train_x, train_y)

# 输出模型的预测效果
result_prob = model.predict({'gyrx_input': trainer.gyr_x_v,
                             'gyry_input': trainer.gyr_y_v,
                             'gyrz_input': trainer.gyr_z_v,
                             'laccx_input': trainer.lacc_x_v,
                             'laccy_input': trainer.lacc_y_v,
                             'laccz_input': trainer.lacc_z_v,
                             'magx_input': trainer.mag_x_v,
                             'magy_input': trainer.mag_y_v,
                             'magz_input': trainer.mag_z_v,
                             'pres_input': trainer.pressure_v,
                         })
y_predict = np.argmax(result_prob, axis=-1)
print("预测", y_predict)
test_y = np.argmax(test_y, axis=-1)
print("真实", test_y)
# 混淆矩阵 矩阵  行：分类 列：分类
cm = sm.confusion_matrix(test_y, y_predict)
plot_confusion_matrix(cm, ['0'])
print("---------------混淆矩阵\n", cm)

cp = sm.classification_report(test_y, y_predict)
print("---------------分类报告\n", cp)
acc = np.sum(y_predict == test_y) / test_y.size
print(acc)




