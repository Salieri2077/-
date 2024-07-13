% 清屏、清工作区、关闭窗口
clc; clear; close all;
% 设置参数
L = 10; % 滤波器阶数
miu = 0.0000005; % 步长参数
len_RxWaveSym1 = 2000; % 训练样本长度
len_RxWaveSym2 = 0; % 均衡样本长度

% 从文件中读取数据
un = load('Inpulse_for_matlab.csv');
un = un(1:len_RxWaveSym1 + len_RxWaveSym2);

% 初始化权重向量和输入序列
wn = zeros(2 * L, 1);
Xn = zeros(1, len_RxWaveSym1); % 输入序列
en = zeros(1, len_RxWaveSym1); % 误差序列
RxWave = un(1:len_RxWaveSym1)';
TxWave1 = un(1:len_RxWaveSym1)';

% 训练过程
for i = L + 1:len_RxWaveSym1-L
    xn = RxWave(L+i-1:-1:i-L)';
    Xn(i) = wn' * xn;
    XXn = xn;
    en(i) = TxWave1(i) - Xn(i);
    wn = wn + (miu * en(i) * XXn')'; % LMS算法更新公式
end

% % 均衡过程
% RxWave_bu = [RxWave; zeros(L, 1)];
% Xn = zeros(1, len_RxWaveSym2); % 重新初始化 Xn
% for i = len_RxWaveSym1 + 1:len_RxWaveSym1 + len_RxWaveSym2
%     xn = RxWave_bu(L+i-1:-1:i-L)';
%     Xn(i-len_RxWaveSym1) = wn' * xn;
% end

% 绘制对比图
figure;
plot(1:len_RxWaveSym1, TxWave1, 'b', 'linewidth', 2); hold on;
plot(1:len_RxWaveSym1, Xn, 'r--', 'linewidth', 2);
xlabel('样本序号');
ylabel('信号值');
title('LMS-真实信号与预测信号对比');
legend('真实信号', '预测信号');
grid on;

compare = abs(Xn-TxWave1);
figure;
plot(compare);
fprintf("MSE: %.4f \n",mean(compare.^2));