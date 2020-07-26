clc;clear all;close all;
filename = '.\crabs.txt';
[SP,SEX,RW,CL]=textread(filename,'%s %s %f %f');

n = length(CL);
S = cell2mat(SEX);
ID = find(S=='M');
id2 = setdiff([1:n]',ID);
figure('NumberTitle', 'off', 'Name', 'ԭʼ���ݷֲ�');
plot(RW(ID),CL(ID),'ro');hold on
plot(RW(id2),CL(id2),'bo');
xlabel('RW'),ylabel('CL')
grid on;hold off

x = [RW,ones(n,1)];
y = CL;
load('crabsdata_B_True.mat');
beta1 = B_True(:,1);beta2 = B_True(:,2);

K = 2; m = 2;
data = [x,y];
stop_epsilon = 1e-6;
iter_max = 100;
dim = 2;
%% Ԥ������һ����ʼ���ع�ϵ��Beta_ini_list
Beta_ini_list = generate_initial_B(x,y,K,iter_max);
% load('tone_rawdata.mat')
% save Beta_ini_list.mat  Beta_ini_list
%% ���ѭ������֤ĳ�㷨�ı���
% ��ͳkmeans����
t1 = tic;
[B_kmeans,B_rela_err] = test_kmeans(data,K,Beta_ini_list,iter_max,beta1,beta2);
t_kmeans = toc(t1)
% kmeans+L1����
t2 = tic;
[B_kmeans_L1,B_rela_err_L1] = test_kmeans_L1(data,K,Beta_ini_list,iter_max,beta1,beta2);
t_kmeansL1 = toc(t2)
% ��ͳFCRM����(m=2)
t3 = tic;
[B_FCRM,B_rela_err_FCRM] = test_FCRM(data,K,Beta_ini_list,iter_max,beta1,beta2);
t_FCRM = toc(t3)
%FCRM+L1����(m=2)
t22 = tic;
[B_FCRM_L1,B_rela_FCRM_L1] = test_FCRM_L1(data,K,Beta_ini_list,iter_max,beta1,beta2);
t_FCRML1= toc(t22);
% FCRMa����(m=4)
Alpha = 0.65;
t4 = tic;
[B_FCRa,B_rela_FCRa,pi1_FCRa] = test_FCRMa(data,4,Alpha,K,Beta_ini_list,iter_max,beta1,beta2,dim);
% [B_FCRa,B_rela_FCRa] = test_FCRMa(data,4,Alpha,K,Beta_ini_list,iter_max,beta1,beta2);
t_FCRa = toc(t4)
% FCRMa+L1����(m=4) 
t5 = tic;
[B_FCRaL1,B_rela_FCRaL1] = test_FCRMaL1(data,4,Alpha,K,Beta_ini_list,iter_max,beta1,beta2);
t_FCRaL1 = toc(t5)
% FCRM+entropy����
entropy_coef = 0.04;
t6 = tic;
[B_FCRME,B_rela_FCRRE] = test_FCRME(data,K,entropy_coef,Beta_ini_list,iter_max,beta1,beta2,dim);
t_FCRME = toc(t6)
% FCRM+entropy+L1����
entropy_coef = 0.04;
t7 = tic;
[B_EnL1,B_rela_EnL1] = test_Entropy_L1(data,K,entropy_coef,Beta_ini_list,iter_max,beta1,beta2,dim);
t_EnL1 = toc(t7)
% ��ͳEM����
t8 = tic;
[B_EM,B_rela_EM] = test_EM(data,K,Beta_ini_list,iter_max,beta1,beta2);
t_EM = toc(t8)
% EM+Laplace����
t9 = tic;
[B_EM_Lap,B_rela_EM_Lap] = test_EM_Lap(data,K,Beta_ini_list,iter_max,beta1,beta2);
t_EM_Lap = toc(t9)
%% plot 200������Ļع�ϵ����������ͼ��beta_0Ϊ�����꣬beta1Ϊ������
% ��ʵ������ƽ��Ļع�ϵ����2,0��,(0,1)
%% plot 
% ԭʼ���ݼ�5���쳣ֵ��3,4��;
figure(1);
plot(stretchratio,tuned,'ko');
hold on;
plot(x_outlier,y_outlier,'k*');
hold off;
xlabel('toned');
ylabel('stretch ratio');
axis([1.2,3.2,1.2,4.2])
% keamns vs kmeans_L1
figure(2);
plot(stretchratio,tuned,'ko');
hold on;
plot(x_outlier,y_outlier,'k*');
b1 = b(1:2);
b2 = b(3:4);
x11 = 1.2:0.01:3.2;
plot(x11,x11*b1(1)+b1(2),'k--');
plot(x11,x11*b2(1)+b2(2),'k--');
b1 = bb(1:2);
b2 = bb(3:4);
plot(x11,x11*b1(1)+b1(2),'k-');
plot(x11,x11*b2(1)+b2(2),'k-');
xlabel('toned');
ylabel('stretch ratio');
axis([1.2,3.2,1.2,4.2])
hold off;
% FCRM vs FCRM_L1
figure(3);
plot(stretchratio,tuned,'ko');
hold on;
plot(x_outlier,y_outlier,'k*');
b1 = b(1:2);
b2 = b(3:4);
x11 = 1.2:0.01:3.2;
plot(x11,x11*b1(1)+b1(2),'k--');
plot(x11,x11*b2(1)+b2(2),'k--');
b1 = bb(1:2);
b2 = bb(3:4);
plot(x11,x11*b1(1)+b1(2),'k-');
plot(x11,x11*b2(1)+b2(2),'k-');
xlabel('toned');
ylabel('stretch ratio');
axis([1.2,3.2,1.2,4.2])
hold off;
% FCRa vs FCRa_L1
figure(4);
plot(stretchratio,tuned,'ko');
hold on;
plot(x_outlier,y_outlier,'k*');
b1 = b(1:2);
b2 = b(3:4);
x11 = 1.2:0.01:3.2;
plot(x11,x11*b1(1)+b1(2),'k--');
plot(x11,x11*b2(1)+b2(2),'k--');
b1 = bb(1:2);
b2 = bb(3:4);
plot(x11,x11*b1(1)+b1(2),'k-');
plot(x11,x11*b2(1)+b2(2),'k-');
xlabel('toned');
ylabel('stretch ratio');
axis([1.2,3.2,1.2,4.2])
hold off;
% FCRE vs FCRE_L1
figure(5);
plot(stretchratio,tuned,'ko');
hold on;
plot(x_outlier,y_outlier,'k*');
b1 = b(1:2);
b2 = b(3:4);
x11 = 1.2:0.01:3.2;
plot(x11,x11*b1(1)+b1(2),'k--');
plot(x11,x11*b2(1)+b2(2),'k--');
b1 = bb(1:2);
b2 = bb(3:4);
plot(x11,x11*b1(1)+b1(2),'k-');
plot(x11,x11*b2(1)+b2(2),'k-');
xlabel('toned');
ylabel('stretch ratio');
axis([1.2,3.2,1.2,4.2])
hold off;
% MixregL vs FCRE_L1
figure(6);
plot(stretchratio,tuned,'ko');
hold on;
plot(x_outlier,y_outlier,'k*');
b1 = b(1:2);
b2 = b(3:4);
x11 = 1.2:0.01:3.2;
plot(x11,x11*b1(1)+b1(2),'k--');
plot(x11,x11*b2(1)+b2(2),'k--');
b1 = bb(1:2);
b2 = bb(3:4);
plot(x11,x11*b1(1)+b1(2),'k-');
plot(x11,x11*b2(1)+b2(2),'k-');
xlabel('toned');
ylabel('stretch ratio');
axis([1.2,3.2,1.2,4.2])
hold off;
%% 
