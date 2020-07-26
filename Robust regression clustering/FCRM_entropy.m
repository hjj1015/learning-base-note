function varargout = FCRM_entropy(data,K,entropy_coef,B_initial)
% goal:使用L1距离测度，加熵项惩罚的模型求解混合回归问题
% {x_i,y_i} from K hyperplanes with parametres B1,B2，...,B_k 
% data:(x_i,y_i),x_i \in R^{1*dim},最后一列全为1
%                y_i \in R 
%                Sample_size = N
%   entropy_coef:熵惩罚项的系数>0，需调节，历史文献entropy_coef=0.9898（0.7，它用的是二范数）
%   B_initial： K个初始的超平面系数，d*K
% 这里跟FCM模型的差别是没有模糊因子m的设置
x = data(:,1:(end-1));
y = data(:,end);
[N,D] = size(x);
threshold = 1e-6;
m = 1;
%% initial values for hyperplanes' parametres
if nargin > 3
    B = B_initial;    
else
    B_initial = init_params(x,y);
    B = B_initial;
end
e_ik = error_ik(x,y,B_initial);
U_ik = fuzzy_u_ik(e_ik,entropy_coef);

%% 停机方式选择，目标函数第j+1次迭代值与第j次迭代目标值的差异,设迭代步数不超过10000
Obj_Iter_Values = zeros(1000,1);  % 预分配内存，存储每步迭代的目标函数值
t = tic;
j = 1 ;  % 迭代次数，不超过10000
while j<1000
    total_residule = calcu_residule(x,y,B,U_ik);
    Obj_Iter_Values(j+1) = total_residule;
    pW = U_ik.^m;

    % new value for parameters of each Component
    Nk = sum(U_ik, 1);   % pW 按列求和，得1*K的矩阵
    pPi = Nk/N;            %这里的pPi也是（1，K）大小,是各簇中样本占总样本量的概率
    shift_X = zeros(N,D,K);
    shift_Y = zeros(N,1,K);
    % 加权最小二乘计算回归系数
    for k =1:K
    shift_X(:,:,k) = x.*repmat(pW(:,k),1,D);   % shift_X(:,:,k)是N*D大小
    shift_Y(:,:,k) = y.*repmat(pW(:,k),1,1);   % shift_Y(:,:,k)是N*1大小
    B(:,k) = (shift_X(:,:,k)'*shift_X(:,:,k))\(shift_X(:,:,k)'*shift_Y(:,:,k));
%     %% ADMM算法求解|y-x*Bk|
%     B(:,k) = lad(shift_X(:,:,k), shift_Y(:,:,k), 1.0, 1);
    end
    % check for convergence 
    if abs(Obj_Iter_Values(j+1) - Obj_Iter_Values(j)) < (threshold)
        disp(['第',num2str(j),'次迭代的总残差前后两步差的绝对值小于1e-12']);
        break;
    end
    %   if code run more than 10s ,stop it whether or not convergent
    if toc(t) >20
        break;
    end 
    j = j + 1;
    e_ik = error_ik(x,y,B);
    U_ik = fuzzy_u_ik(e_ik,entropy_coef);
end

Obj_Iter_Values = Obj_Iter_Values(Obj_Iter_Values~=0);

if nargout == 1
    varargout = {U_ik};
else
    model = [];
    model.B = B;
    model.Beta_initial = B_initial;
    model.pPi = pPi;
    model.Obj_Iter_Values = Obj_Iter_Values;
    varargout = {U_ik, model};
    disp('total_residule');
    disp(total_residule);
    disp('total_av_residule');
    disp(total_residule/N);
end

%% 参数初始化
function B = init_params(X,Y)    

num = fix(N/K);
rnum = N-num*K;

xx = zeros(num,D,K);
yy = zeros(num,1,K);
x2 = zeros(num+rnum,D,K);
y2 = zeros(num+rnum,1,K);

B = zeros(D,K);
rndp = randperm(N);
X = X(rndp,:);
Y = Y(rndp,:);
for kk = 1:(K-1)
    xx(:,:,kk) = X(((kk-1)*num+1):(kk*num),:);
    yy(:,:,kk) = Y(((kk-1)*num+1):(kk*num),:);
    % 随机等分K类，最小二乘法初始化K个回归系数
    B(:,kk) = (xx(:,:,kk)'*xx(:,:,kk))\(xx(:,:,kk)'*yy(:,:,kk));
end
% where kk == K 
x2(:,:,K) = X(((K-1)*num+1):end,:);
y2(:,:,K) = Y(((K-1)*num+1):end,:);
B(:,K) = (x2(:,:,K)'*x2(:,:,K))\(x2(:,:,K)'*y2(:,:,K));

end


%% 计算模糊矩阵U_ik
function U_ik = fuzzy_u_ik(e_ik,penalty_coef)
% 输入计算好的 ：e_ik ，第i样本到第k超平面的误差平方,N*K
% U_ik ：模糊矩阵U_ik ,N*K
%
dis_lambda = -e_ik/penalty_coef;
% 计算exp(-|dij|/penalty_coef),exp_dis : size,N*K
U_ik = zeros(N,K);
for i=1:N
    for kk=1:K
        et = -dis_lambda(i,kk);
        rowt = dis_lambda(i,:) + et;
        exp_rowt = exp(rowt);
        U_ik(i,kk) = 1/sum(exp_rowt);
    end
end
% exp_dis = exp(dis_lambda);
% sum_exp = sum(exp_dis,2); % 按列对exp_dis求和，N*1
% rep_exp = repmat(sum_exp,1,K);
% % 计算最终的模糊矩阵
% U_ik = exp_dis./rep_exp;

end
%% 计算e_ik矩阵，第i样本到簇k的误差
function e_ik = error_ik(X,Y,Beta)
% Beta: K个超平面系数，d*K
% e_ki:第i样本到簇k的误差平方,N*K
% 
Y_K = repmat(Y, 1, K); 
XB = X*Beta;
gap = Y_K - XB;
e_ik = abs(gap);  % 第i样本到簇k的误差,L1范数
e_ik = e_ik.*e_ik;% 第i样本到簇k的误差,L2范数
end

function total_residule = calcu_residule(x,y,B,U_ik)
% 计算总的残差平方和
resid = zeros(K,1);
for jj = 1:K
    Bk = B(:,jj);
    pW_k = diag(U_ik(:,jj)).^m;
    inter_dis = pW_k*abs(y-x*Bk);  % N*1，样本绝对值误差乘以权重
    %%%%%%%%%%%%%%%%%%%%%%%%%%% 可将pGamma_k先不加权计算，看看结果
    resid(jj) = sum(inter_dis);     
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
end
total_residule = sum(resid);  % 计算所有样本到所有平面的残差和
end

end