clc;clear all;
%% !!!!!
% ��Ҫ���������ѡ������ϵ������ֵ��֤����֤ԭ���ݷ���Խ��ϵ��rԽ�󣬳������Եģ�ͬʱ�����ݼ����������쳣ֵԽ��
% ����ҲԽ��̬��rҲӦ�ø���
% �¸�ʵ���������߼Ӹ߸ܸ��쳣ֵ�ıȽ�
%% ��������
n = 100;  % ��������
% ����ǿ��0.1��0.1
pow_noise = [0.3,0.3];
% fprintf('������ʵƽ��ֱ����ǿ�ȣ�%4.2f\t%4.2f\n',pow_noise(1),pow_noise(2));
% �������ͣ���˹��������˹����
noise_c = ['N','N'];  %N U L T P ��5������������P��ʾ����Ⱦ�ĸ�˹����
% fprintf('������ʵƽ����������ͣ�%4s\t%4s\n',noise_c(1),noise_c(2));
seednum = 10;
%% 
[x1,y1,beta1,x2,y2,beta2,true_residul] = ge_lines2(n,pow_noise,noise_c,seednum);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure(1)
% plot(x1(:,1),y1,'kO');
% hold on
% plot(x2(:,1),y2,'kO');
% % grid on
% xlim([-2,2]);hold off
% title('old method')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  �ϲ���������Ϊ x��y,�������ɢ����
x = [x1;x2]; y = [y1;y2];
% %% ��Ӹ߸ܸ��쳣ֵ
% n_outlier = fix(n*0.05);
% x_o = repmat([2,1],n_outlier,1);
% y_o = 10*ones(n_outlier,1);
% x = [x;x_o];
% y = [y;y_o];
% rnd_indx = randperm(n+n_outlier);

% %% ����ȡ����
% mouse_data = ginput;
% %%
% figure(2)
% plot(x1(:,1),y1,'ko');
% hold on
% plot(x2(:,1),y2,'ko');
% plot(mouse_data(:,1),mouse_data(:,2),'k*');
% % grid on
% xlim([-2,2]);hold off
%% 
% [n_outier,~] = size(mouse_data);
% data3 = [mouse_data(:,1),ones(n_outier,1),mouse_data(:,2)];
% data = [data;data3];
data = [x,y];
%% ��ɢ����
% rnd_indx = randperm(n+n_outier);
rnd_indx = randperm(n);
data = data(rnd_indx,:);
x = data(:,1:(end-1));
y = data(:,end);
%% 
B_True = [beta1,beta2];
K = 2; m = 2;
stop_epsilon = 1e-6;
iter_max = 100;
dim = 2;
%% Ԥ������һ����ʼ���ع�ϵ��Beta_ini_list
Beta_ini_list = generate_initial_B(x,y,K,iter_max);
% Beta_ini_list = randn(6,iter_max);
%% ���ѭ������֤ĳ�㷨�ı���
% FCRM+entropy����
% entropy_coef = 0.02;
% t6 = tic;
% [B_FCRME,B_rela_FCRRE,pi1_FCRE] = test_FCRME(data,K,entropy_coef,Beta_ini_list,20,beta1,beta2,dim);
% % [B_FCRME,B_rela_FCRRE,pi1_FCRE] = test_FCRME(data,K,entropy_coef,B_kmeans',iter_max,beta1,beta2,dim);
% t_FCRME = toc(t6)
% FCRM+entropy+L1����
% t7 = tic;
% entropy_coef = 0.11;% 1����׼��
% [B_EnL1,B_rela_EnL1,pi1_EnL1] = test_Entropy_L1(data,K,entropy_coef,Beta_ini_list,10,beta1,beta2,dim);
% % [B_EnL1,B_rela_EnL1,pi1_EnL1] = test_Entropy_L1(data,K,entropy_coef,B_kmeans',iter_max,beta1,beta2,dim);
% t_EnL1 = toc(t7)
%% 
% t3 = tic;
% [B_FCRM,B_rela_err_FCRM,pi1_FCRM] = test_FCRM(data,K,Beta_ini_list,20,beta1,beta2,dim);
% % [B_FCRM,B_rela_err_FCRM,pi1_FCRM] = test_FCRM(data,K,B_kmeans',iter_max,beta1,beta2,dim);
% t_FCRM = toc(t3)
%% 
% entr = 0.01:0.01:0.4;
entr = 0.1:0.01:0.5;   
nn = length(entr);
BS = zeros(iter_max,nn);
AE_B = zeros(iter_max,nn);
% ����r����õĻع�ϵ��
B_En_best = zeros(iter_max,2*dim);
for i = 1:nn
    [B_EnL1,B_rela_EnL1,~] = test_Entropy_L1(data,K,entr(i),Beta_ini_list,iter_max,beta1,beta2,dim);
    AE_B(:,i) = cal_cu_AE(B_EnL1,beta1,beta2);
    % ����ͬһr�£�100���������\beta������
    BS(:,i) = B_rela_EnL1;
    [~,id_] = min(B_rela_EnL1);
    B_En_best(i,:) = B_EnL1(id_,:);
end
BS = sort(BS);
AE_B = sort(AE_B);
% ������b,�����
b = BS(1,:);
% �������b2,�����
b2 = AE_B(1,:);
sucess_n = zeros(1,nn);
for i=1:nn
    % ������С��1%��Ϊ���Ƴɹ�
    sucess_n(i) = length(find(BS(:,i)>pow_noise(1)));  % 0.04
end
sucess_rate = 1 - sucess_n/100;

save entropy_r_choice.mat
%% ���ĳ�γ�ʼ��������
% entr2 = (pow_noise(1)-0.05):0.01:(pow_noise(1)+0.05);
% id_s = find(entr==entr2(1));
% id_e = find(entr==entr2(end));
% 
% 
% [~,id1] = min(b);
% x0 = entr(id1);
% 
% b_ = b(id_s:id_e);
% [~,id11] = min(b_);
% x4 = entr2(id11);
% 
% 
% [~,id2] = min(b2);
% x1 = entr(id2);
% id3 = find(sucess_rate==1);
% x3 = entr(id3(1));
% 
% b2_ = b2(id_s:id_e);
% [~,id22] = min(b2_);
% x5 = entr2(id22);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure(1)
% subplot(3,1,1)
% plot(entr,b2,'k.-')
% hold on
% plot([x1,x1], get(gca, 'YLim'), '-r', 'LineWidth', 1) % ��ɫ�����Ϊ1
% text(x1,b2(abs(entr-x1)<1e-3)+0.005,['(' num2str(x1) ',' num2str(b2(abs(entr-x1)<1e-3)) ')'])
% plot([x5,x5], get(gca, 'YLim'), '-b', 'LineWidth', 1) % ��ɫ�����Ϊ1
% text(x5,b2(abs(entr-x5)<1e-3)+0.005,['(' num2str(x5) ',' num2str(b2(abs(entr-x5)<1e-3)) ')'])
% plot([x3,x3], get(gca, 'YLim'), '-r', 'LineWidth', 1) % ��ɫ�����Ϊ3
% text(x3,b2(abs(entr-x3)<1e-3)+0.005,['(' num2str(x3) ',' num2str(b2(abs(entr-x3)<1e-3)) ')']);
% hold off
% xlabel('parameter \gamma of entropy term')
% ylabel('Ae($\hat{\beta}$)','Interpreter','latex')
% subplot(3,1,2)
% plot(entr,b,'k.-')
% hold on
% plot([x0,x0], get(gca, 'YLim'), '-r', 'LineWidth', 1) % ��ɫ�����Ϊ3
% text(x0,b(abs(entr-x0)<1e-3)+0.005,['(' num2str(x0) ',' num2str(b(abs(entr-x0)<1e-3)) ')']);
% plot([x4,x4], get(gca, 'YLim'), '-b', 'LineWidth', 1) % ��ɫ�����Ϊ3
% text(x4,b(abs(entr-x4)<1e-3)+0.005,['(' num2str(x4) ',' num2str(b(abs(entr-x4)<1e-3)) ')']);
% plot([x3,x3], get(gca, 'YLim'), '-r', 'LineWidth', 1) % ��ɫ�����Ϊ3
% text(x3,b(abs(entr-x3)<1e-3)+0.005,['(' num2str(x3) ',' num2str(b(abs(entr-x3)<1e-3)) ')']);
% hold off
% % title('Relative error of parameter \beta')
% xlabel('parameter \gamma of entropy term')
% ylabel('Re($\hat{\beta}$)','Interpreter','latex')
% subplot(3,1,3)
% plot(entr,sucess_rate,'k.-')
% hold on
% plot([x0,x0], get(gca, 'YLim'), '-r', 'LineWidth', 1) % ��ɫ�����Ϊ3
% text(x0+0.01,sucess_rate(abs(entr-x0)<1e-3),['(' num2str(x0) ',' num2str(sucess_rate(abs(entr-x0)<1e-3)) ')']);
% plot([x4,x4], get(gca, 'YLim'), '-b', 'LineWidth', 1) % ��ɫ�����Ϊ3
% text(x4+0.01,sucess_rate(abs(entr-x4)<1e-3),['(' num2str(x4) ',' num2str(sucess_rate(abs(entr-x4)<1e-3)) ')']);
% plot([x3,x3], get(gca, 'YLim'), '-r', 'LineWidth', 1) % ��ɫ�����Ϊ3
% text(x3,sucess_rate(abs(entr-x3)<1e-3)-0.01,['(' num2str(x3) ',' num2str(sucess_rate(abs(entr-x3)<1e-3)) ')']);
% hold off
% % title('Relative error of parameter \beta')
% xlabel('parameter \gamma of entropy term')
% ylabel('stablity of $\hat{\beta}$ ','Interpreter','latex')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ����ͼ����  clc;clear all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(2)
plot(entr,sucess_rate,'k.-')
x1 = 1; % 0.1��0.11  
hold on
plot([x1,x1], get(gca, 'YLim'), '-b', 'LineWidth', 1) % ��ɫ�����Ϊ1
text(x1,sucess_rate(abs(entr-x1)<1e-3)-0.01,['(' num2str(x1) ',' num2str(sucess_rate(abs(entr-x1)<1e-3)) ')'])
xlabel('parameter \gamma of entropy term')
ylabel('stabity of $\hat{\beta}$ ','Interpreter','latex')
grid on

figure(3)
subplot(2,1,1)
plot(entr,b2,'k.-'),grid on
hold on
plot([x1,x1], get(gca, 'YLim'), '-b', 'LineWidth', 1) % ��ɫ�����Ϊ1
text(x1,b2(abs(entr-x1)<1e-3)+0.003,['(' num2str(x1) ',' num2str(b2(abs(entr-x1)<1e-3)) ')'])
xlabel('parameter \gamma of entropy term')
ylabel('Ae($\hat{\beta}$)','Interpreter','latex')

subplot(2,1,2)
plot(entr,b,'k.-')
hold on
plot([x1,x1], get(gca, 'YLim'), '-b', 'LineWidth', 1) % ��ɫ�����Ϊ1
text(x1,b(abs(entr-x1)<1e-3)+0.002,['(' num2str(x1) ',' num2str(b(abs(entr-x1)<1e-3)) ')'])
xlabel('parameter \gamma of entropy term')
ylabel('Re($\hat{\beta}$)','Interpreter','latex')
grid on
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% figure(2)
% subplot(2,1,1)
% plot(s,b,'k.-')
% hold on
% plot([x0,x0], get(gca, 'YLim'), '-b', 'LineWidth', 1) % ��ɫ�����Ϊ3
% hold on
% plot([0.32,0.32], get(gca, 'YLim'), '-r', 'LineWidth', 1) % ��ɫ�����Ϊ3
% hold off
% title('Relative error of parameter \beta')
% xlabel('parameter \gamma of entropy term')
% subplot(2,1,2)
% plot(s,b2,'k.-')
% hold on
% plot([x0,x0], get(gca, 'YLim'), '-b', 'LineWidth', 1) % ��ɫ�����Ϊ1
% hold on
% plot([0.32,0.32], get(gca, 'YLim'), '-r', 'LineWidth', 1) % ��ɫ�����Ϊ1
% hold off
% xlabel('parameter \gamma of entropy term')
% title('Numerical stability')
