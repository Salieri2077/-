import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# setting = 'inpulse3' # 第一次informer但是没存结果
# setting = 'inpulse_informer_24'
setting = 'inpulse_informer_24_real'
# setting = 'inpulse_informer_48'
# setting = 'inpulse_informer_96'

hss = np.load('./results/'+setting+'/metrics.npy')
preds = np.load('./results/'+setting+'/pred.npy')
trues = np.load('./results/'+setting+'/true.npy')
a = trues[0,0,0]
# # [samples, pred_len, dimensions]
# preds.shape, trues.shape
num_tau = preds.shape[0] # 窗体移动次数-384
num_time = preds.shape[2] # 不同时间的-20个不同信道
num_pred = preds.shape[1] # 向下预测的信道冲激响应-24
origin_data = np.zeros((num_tau+num_pred, preds.shape[2]))
pred_data = np.zeros((num_tau+num_pred, preds.shape[2]))
# origin = # num_time
for i in range(num_tau):
    if i == 0:
        origin_data[:num_pred,:] = trues[i,:num_pred,:]
        pred_data[:num_pred,:] = preds[i,:num_pred,:]
    else:
        origin_data[num_pred+i,:] = trues[i,-1,:] # 窗体每次移动1个单位
        pred_data[num_pred+i,:] = preds[i,-1,:]
origin_data = origin_data.T
pred_data = pred_data.T
#
plt.figure(figsize=(10, 6))
plt.plot(origin_data.T, label='GroundTruth')
plt.plot(pred_data.T, label='Informer-Prediction')
plt.legend()
plt.show()
# 绘制 origin_data 的伪彩图
plt.figure(figsize=(10, 6))
plt.imshow(origin_data, aspect='auto', cmap='viridis')
plt.colorbar(label='Value')
plt.title('Origin Data Pseudocolor Plot')
plt.xlabel('Delay')
plt.ylabel('Time')
plt.show()
# 绘制 pred_data 的伪彩图
plt.figure(figsize=(10, 6))
plt.imshow(pred_data, aspect='auto', cmap='viridis')
plt.colorbar(label='Value')
plt.title('Informer-Predicted Data Pseudocolor Plot')
plt.xlabel('Delay')
plt.ylabel('Time')
plt.show()

# # draw OT prediction
# plt.figure()
# plt.plot(trues[-1,:,-1], label='GroundTruth')
# plt.plot(preds[-1,:,-1], label='Prediction')
# plt.legend()
# plt.show()

# # draw HUFL prediction
# plt.figure()
# plt.plot(trues[-1,:,0], label='GroundTruth')
# plt.plot(preds[-1,:,0], label='Prediction')
# plt.legend()
# plt.show()

# # 读取CSV文件
# data = pd.read_csv('./data/ETT/ETTh1.csv')

# # 提取OT列的数据的17397行到17421行的数据
# ot_data = data['OT'].iloc[17396:17421]

# # 创建图像并绘制曲线
# plt.figure(figsize=(10, 6))
# plt.plot(ot_data, label='OT')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.title('OT Data')
# plt.legend()
# plt.grid(True)
# plt.show()
