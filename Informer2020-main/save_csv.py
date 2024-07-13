import numpy as np
import pandas as pd
import scipy.io

# # 读取 .npy 文件
# data = np.load(r"C:\Users\Lenovo\Downloads\Informer2020-main\results\informer_ETTh1_ftM_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0\pred.npy")
# # 循环保存数据到xlsx文件
# num_experiments = data.shape[0]
# num_sheets_per_file = 24

# for i in range(0, num_experiments, num_sheets_per_file):
#     # 创建一个Excel写入对象
#     writer = pd.ExcelWriter(f'experiment_results_{i//num_sheets_per_file}.xlsx', engine='xlsxwriter')
    
#     # 将每次实验结果写入一个Sheet
#     for j in range(num_sheets_per_file):
#         sheet_name = f'Sheet{j+1}'
#         # 创建DataFrame
#         df = pd.DataFrame(data[i+j], columns=[f'Data_{k+1}' for k in range(7)])
#         # 将DataFrame写入Excel
#         df.to_excel(writer, sheet_name=sheet_name, index=False)
    
#     # 关闭ExcelWriter对象
#     writer.close()

mat = scipy.io.loadmat('./data/ETT/NOF1_001.mat')
y = np.array(mat['h'])
# 变成一列--单变量预测
num_row_time = 5
y = np.abs(y[:num_row_time,:])
y = y.reshape(1,-1)
y = pd.DataFrame(y.T)
y.rename(columns={y.columns[0]: "together"}, inplace=True)
# 生成时间标签并添加到DataFrame索引
start_date = pd.to_datetime('2016-07-01 00:00')
time_labels = pd.date_range(start=start_date, periods=y.shape[0], freq='H')
y.index = time_labels
# 重命名索引为'date'
y.index.name = 'date'
# 保存DataFrame为CSV文件
y.to_csv('./data/ETT/Inpulse_hour.csv')
print('over')