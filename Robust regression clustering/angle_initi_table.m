function angle_Total = angle_initi_table(n1,n2,dim)

% ����ǿ��0.1��0.1
pow_noise = [0.1,0.1];
% �������ͣ���˹��������˹����
noise_c = [1,1];
%% ��������
[x1,y1,beta1,x2,y2,beta2] = gen_data(n1,n2,dim,pow_noise,noise_c,0);
%%  �ϲ���������Ϊ x��y,�������ɢ����
x = [x1;x2]; y = [y1;y2];
rnd_indx = randperm(n1+n2);
x = x(rnd_indx,:);
y = y(rnd_indx,:);
%% �����ʼ��beta_0
% beta_initial = initial_beta(x,y);
%% 
inter_angle = angle_calcu(beta1,beta2); %��ʵƽ��ļн�
beta_init_one = (x'*x) \ (x'*y);
inter_angle1 = angle_calcu(beta1,beta_init_one); 
inter_angle2 = angle_calcu(beta2,beta_init_one); 
angle_Total = [inter_angle,inter_angle1,inter_angle2];