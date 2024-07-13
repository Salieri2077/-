clc, clear, close all; % 清屏、清工作区、关闭窗口
un = load('Inpulse_for_matlab.csv'); % 从文件中读取数据
un = un(1:2048);
N = length(un); % 信号观测长度
warning off; % 消除警告
feature jit off; % 加速代码执行
a1 = 0.99; % 一阶 AR 参数
% 产生期望响应信号和观测数据矩阵
n0 = 1; % 虚实现 n0 步线性预测
M = 2; % 滤波器阶数
b = un(n0+1:N); % 预测的期望响应
L = length(b); 
un1 = [zeros(M-1, 1)', un']; % 扩展数据
A = zeros(M, L); 
for k = 1:L 
    A(:, k) = un1(M-1+k:-1:k); % 构建观测数据矩阵
end 
% 应用 RLS 算法进行迭代寻优计算最优权向量
delta = 0.004; % 调整参数
lamda = 1.5; % 遗忘因子
w = zeros(M, L+1); 
epsilon = zeros(L, 1); 
P1 = eye(M) / delta; 
% RLS 迭代算法过程
for k = 1:L 
    PIn = P1 * A(:, k); 
    denok = lamda + A(:, k)' * PIn; 
    kn = PIn / denok; 
    epsilon(k) = b(k) - w(:, k)' * A(:, k); 
    w(:, k+1) = w(:, k) + kn * conj(epsilon(k)); 
    P1 = P1 / lamda - kn * A(:, k)' * P1 / lamda; 
end 
w1 = w(1, :); 
w2 = w(2, :); 
MSE = abs(epsilon).^2; 
MSE_P = MSE; 

W1 = w1';
W2 = w2';
figure, plot(1:L, MSE_P * ones(1, L), 'r', 'linewidth', 2), title(' MSE'); grid on; 
figure, plot(1:length(W1), W1 * ones(1, length(W1)), 'r', 'linewidth', 2), title('权值'); hold on; 
plot(1:length(W2), W2 * ones(1, length(W2)), 'b', 'linewidth', 2), title('权值'); hold on; 
grid on; legend('\alpha1=0', '\alpha2=-1');
% 生成预测值
predicted_values = zeros(size(b));
for k = 1:L
    predicted_values(k) = w(:, k)' * A(:, k);
end
% 绘制对比图
figure;
plot(1:N, un, 'b', 'linewidth', 2); hold on;
plot(n0+1:N, predicted_values, 'r--', 'linewidth', 2);
xlabel('样本序号');
ylabel('信号值');
title('真实信号与预测信号对比');
legend('真实信号', '预测信号');
grid on;

compare = abs(predicted_values-un(1:end-1));
figure;
plot(compare);
fprintf("MSE: %.4f \n",mean(compare.^2));