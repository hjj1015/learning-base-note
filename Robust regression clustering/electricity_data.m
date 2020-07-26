clc;clear all;close all;
filename = '.\electricity_data.txt';
[y,X1,X2,X3]=textread(filename,'%f %f %f %f');

n = length(y);
x = [X1,X2,X3,ones(n,1)];

% load('electricity_data_B_True.mat');
% beta1 = B_True(:,1);beta2 = B_True(:,2);
K = 5;
m = 2;
[~,dim] = size(x);
beta_initial = randn(dim,K);
Beta_inital = initial_B_strategy(x,y,K);
%% 调用FCRM算法求解beta_hat
fprintf('############## FCRM求回归聚类任务结果输出 ##############\n');
[U_FCRM,FCRM_model] = FCRM_solve_K_plane(x,y,K,m,Beta_inital);%% 识别估计出的回归系数B，哪个是第一个平面的系数
B_hat_FCRM = FCRM_model.B;
pPi = FCRM_model.pPi;
fprintf('############## FCRM求回归聚类任务结果输出 ##############\n');
%% L1_FCRM 算法求解 ：L1_FCRM_output = L1_FCRM(data,K,m);
data = [x,y];
entropy_coef = 0.03;
fprintf('############## L1_entropy求回归聚类任务结果输出 ##############\n');
% 以相同的初始化beta_initial；或以FCRM算法的结果作为L1_FCRM算法的初始值
[U_FCRM2,L1_FCRM_output] = L1_entropy(data,K,entropy_coef,Beta_inital);
fprintf('############## L1_entropy求回归聚类任务结果输出 ##############\n');
% L1_FCRM算法求解结果分析
B_hat_L1 = L1_FCRM_output.B;
pPi_L1_FCRM = L1_FCRM_output.pPi;