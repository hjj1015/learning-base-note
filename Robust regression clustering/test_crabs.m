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
grid on;hold off

x = [RW,ones(n,1)];
y = CL;
beta_initial = initial_B(x,y,2);
data = [x,y];
stop_epsilon = 1e-10;
%% RL_FCRM�㷨���
% initial.c = 6;
% initial.B = randn(2,initial.c); % d*c ,c����ʼ�ع�ϵ��
% RL_FCRM_output = RL_FCRM(data,stop_epsilon,initial);
% #### ���´���Ŀ��ʱ�򣬻����ն�������Ĵ�ֱ�ӵ�1����
% #### ����alpha�����У�alpha����Ϊ��ֵ����������������������
% RL_FCRM_output = RL_FCRM(data,stop_epsilon);
% B_RL_FCRM = RL_FCRM_output.B;
%% L1_FCRM �㷨��� ��L1_FCRM_output = L1_FCRM(data,K,m,beta_initial);��beta_initialҲ��
fprintf('############## FCRM���ع�������������� ##############\n');
[U_FCRM2,L1_FCRM_output] = L1_FCRM(data,2,2,beta_initial);
B_hat_L1_FCRM = L1_FCRM_output.B;
B1_L1 = B_hat_L1_FCRM(:,1);B2_L1 = B_hat_L1_FCRM(:,2);
[label,data1,data2] = clust_label(x,y,B1_L1,B2_L1);
fprintf('############## L1_FCRM���ع�������������� ##############\n');
figure('NumberTitle', 'off', 'Name', 'L1_FCRMԤ��ĳ�ƽ��طֲ�');
plot(RW(ID),CL(ID),'ro');hold on
plot(RW(id2),CL(id2),'bo');
plot(data1(:,1),data1(:,2),'k+',data2(:,1),data2(:,2),'k*')
grid on
hold off;
%%
% %% ����EM�㷨���beta_hat
% [Px_EM,EM_model] = EM_solve_K_plane(x,y,2,beta_initial);  % ���ø�ʽ��varargout = EM_solve_K_plane(X,Y,K,B_initial)
% B_initial = EM_model.Beta_initial;
% % EM�㷨���������
% B_hat_EM = EM_model.B;
% pPi_EM = EM_model.pPi;
% W_EM = EM_model.pGamma;
% B1_EM = B_hat_PAE(:,1);B2_EM = B_hat_PAE(:,2);
% [label,data1,data2] = clust_label(x,y,B1_EM,B2_EM);
% figure('NumberTitle', 'off', 'Name', 'EMԤ��ĳ�ƽ��طֲ�');
% plot(data1(:,1),data1(:,2),'ro',data2(:,1),data2(:,2),'k^')
% grid on

%% ����PAE�㷨���beta_hat
fprintf('############## PAE���ع�������������� ##############\n');
[Px_PAE,PAE_model] = PAE_solve_K_plane(x,y,2,beta_initial);% ���ø�ʽ��varargout = PAE_solve_K_plane(X,Y,K,B_initial)
% [Px_PAE,PAE_model] = PAE_solve_K_plane(x,y,2,beta_initial);% ���ø�ʽ��varargout = PAE_solve_K_plane(X,Y,K,B_initial)
fprintf('############## PAE���ع�������������� ##############\n');
%% ʶ����Ƴ��Ļع�ϵ��B���ĸ��ǵ�һ��ƽ���ϵ��
B_hat_PAE = PAE_model.B;
pPi = PAE_model.pPi;
W_PAE = PAE_model.pW;
B1_PAE = B_hat_PAE(:,1);B2_PAE = B_hat_PAE(:,2);

[labe2,data1,data2] = clust_label(x,y,B1_PAE,B2_PAE);
figure('NumberTitle', 'off', 'Name', 'PAEԤ��ĳ�ƽ��طֲ�');
plot(RW(ID),CL(ID),'ro');hold on
plot(RW(id2),CL(id2),'bo');
plot(data1(:,1),data1(:,2),'k+',data2(:,1),data2(:,2),'k*')
grid on
hold off;

%% ����FCRM�㷨���beta_hat
m = 2;
fprintf('############## FCRM���ع�������������� ##############\n');
[U_FCRM,FCRM_model] = FCRM_solve_K_plane(x,y,2,m,beta_initial);%% ʶ����Ƴ��Ļع�ϵ��B���ĸ��ǵ�һ��ƽ���ϵ��
fprintf('############## FCRM���ع�������������� ##############\n');
B_hat_FCRM = FCRM_model.B;
pPi = FCRM_model.pPi;
W_FCRM = U_FCRM;
B1_FCRM = B_hat_FCRM(:,1);B2_FCRM = B_hat_FCRM(:,2);

[label3,data1,data2] = clust_label(x,y,B1_FCRM,B2_FCRM);
figure('NumberTitle', 'off', 'Name', 'FCRMԤ��ĳ�ƽ��طֲ�');
plot(RW(ID),CL(ID),'ro');hold on
plot(RW(id2),CL(id2),'bo');
plot(data1(:,1),data1(:,2),'k+',data2(:,1),data2(:,2),'k*')
grid on;
hold off;