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
%% ����FCRM�㷨���beta_hat
fprintf('############## FCRM��ع������������� ##############\n');
[U_FCRM,FCRM_model] = FCRM_solve_K_plane(x,y,K,m,Beta_inital);%% ʶ����Ƴ��Ļع�ϵ��B���ĸ��ǵ�һ��ƽ���ϵ��
B_hat_FCRM = FCRM_model.B;
pPi = FCRM_model.pPi;
fprintf('############## FCRM��ع������������� ##############\n');
%% L1_FCRM �㷨��� ��L1_FCRM_output = L1_FCRM(data,K,m);
data = [x,y];
entropy_coef = 0.03;
fprintf('############## L1_entropy��ع������������� ##############\n');
% ����ͬ�ĳ�ʼ��beta_initial������FCRM�㷨�Ľ����ΪL1_FCRM�㷨�ĳ�ʼֵ
[U_FCRM2,L1_FCRM_output] = L1_entropy(data,K,entropy_coef,Beta_inital);
fprintf('############## L1_entropy��ع������������� ##############\n');
% L1_FCRM�㷨���������
B_hat_L1 = L1_FCRM_output.B;
pPi_L1_FCRM = L1_FCRM_output.pPi;