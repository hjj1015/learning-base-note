function angle_Total = angle_initi_table(n1,n2,dim)

% 加噪强度0.1，0.1
pow_noise = [0.1,0.1];
% 噪声类型：高斯噪声，高斯噪声
noise_c = [1,1];
%% 生成数据
[x1,y1,beta1,x2,y2,beta2] = gen_data(n1,n2,dim,pow_noise,noise_c,0);
%%  合并两类数据为 x，y,并随机打散数据
x = [x1;x2]; y = [y1;y2];
rnd_indx = randperm(n1+n2);
x = x(rnd_indx,:);
y = y(rnd_indx,:);
%% 随机初始化beta_0
% beta_initial = initial_beta(x,y);
%% 
inter_angle = angle_calcu(beta1,beta2); %真实平面的夹角
beta_init_one = (x'*x) \ (x'*y);
inter_angle1 = angle_calcu(beta1,beta_init_one); 
inter_angle2 = angle_calcu(beta2,beta_init_one); 
angle_Total = [inter_angle,inter_angle1,inter_angle2];