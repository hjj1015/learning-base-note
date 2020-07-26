function varargout = EM_solve_K_plane(X,Y,K,B_initial)
%   X :(n1+n2)*dim
%   Y:(n1+n2)*1
%   {x_i,y_i} from K hyperplanes with parametres B1,B2，...,B_k 
%   i = 1,2,...(n1+n2)
%   K个超平面
%   B_initial： K个初始的超平面系数，d*K

% 停机误差限
threshold = 1e-6; 

% 获取样本量N，样本维度D
[N, D] = size(X);
Sigma = zeros(K,1);
%% initial values for hyperplanes' parametres
if nargin > 3
    pPi = zeros(1,K);
    B = B_initial;
    dist = abs(repmat(Y,1,K) - X*B);
    [~,ik] = min(dist,[],2);
    for j = 1:K
        yj = Y(ik==j,:);
        xj = X(ik==j,:);
        Xk = yj - xj*B(:,j);
        nj = length(yj);
        Sigma(j,1) = (Xk'*Xk)/nj;
        pPi(j) = nj/N;
    end
    pGamma = pPi;
else
    [B,pPi] = init_params();
    B_initial = B;
end

%% 停机方式选择，目标函数第j+1次迭代值与第j次迭代目标值的差异,设迭代步数不超过10000
Obj_Iter_Values = zeros(1000,1);  % 预分配内存，存储每步迭代的目标函数值

t = tic;
j = 1 ;  % 迭代次数，不超过10000

while j<1000
    [Px,total_residule] = calc_prob();
    Obj_Iter_Values(j+1) = total_residule;
    
    pGamma = Px .* repmat(pPi, N, 1);  % pGamma：N*K   
    % pGamma是N*K隶属度矩阵，第N个样本属于第K个簇的隶属度
    pGamma = pGamma ./ repmat(sum(pGamma, 2), 1, K);  

    % new value for parameters of each Component
    Nk = sum(pGamma, 1);   % pGamma 按列求和，得1*K的矩阵
    pPi = Nk/N;            %这里的pPi也是（1，K）大小,是各簇中样本占总样本量的概率
    shift_X = zeros(N,D,K);
    shift_Y = zeros(N,1,K);
    Sigma =  zeros(K,1); %k类的sigma方差
    for k =1:K
    shift_X(:,:,k) = X.*repmat(pGamma(:,k),1,D);
    shift_Y(:,:,k) = Y.*repmat(pGamma(:,k),1,1);
    %！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    B(:,k) = (shift_X(:,:,k)'*shift_X(:,:,k))\(shift_X(:,:,k)'*shift_Y(:,:,k));
    %！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    % 更新第k类的sigma2方差
    Xt = Y - X*B(:,k);
    Sigma(k,1) = (Xt' * ...
        (diag(pGamma(:, k)) * Xt)) / Nk(k);
    end
    % check for convergence 
    if abs(Obj_Iter_Values(j+1) - Obj_Iter_Values(j)) < (threshold)
        disp(['第',num2str(j),'次迭代的总残差前后两步差的绝对值小于1e-12']);
        break;
    end
%     total_residule_before = total_residule ;
%     B_1_before = B_1;
    %   if code run more than 10s ,stop it whether or not convergent
    if toc(t) > 20
        break;
    end 
    j = j + 1;
end
Obj_Iter_Values = Obj_Iter_Values(Obj_Iter_Values~=0);

if nargout == 1
    varargout = {Px};
else
    model = [];
    model.B = B;
    model.Beta_initial = B_initial;
%     model.B1 = B(:,1);
%     model.B2 = B(:,2);
%     [x11,y11,x22,y22] = USE_coeffi_TO_CLASSIFY_20(X,Y,B(:,1),B(:,2));
%     model.B1 = (x11'*x11)\(x11'*y11);
%     model.B2 = (x22'*x22)\(x22'*y22);
    model.pPi = pPi;
    model.pGamma = pGamma;
    model.Obj_Iter_Values = Obj_Iter_Values;
    varargout = {Px, model};
    disp('total_residule');
    disp(total_residule);
    disp('total_av_residule');
    disp(total_residule/N);
end

%% 参数初始化
function [B,pPi] = init_params()
pPi = 1/K*ones(1,K);     
num = fix(N/K);
rnum = N-num*K;

x = zeros(num,D,K);
y = zeros(num,1,K);
xx = zeros(num+rnum,D,K);
yy = zeros(num+rnum,1,K);

B = zeros(D,K);
rndp = randperm(N);
X = X(rndp,:);
Y = Y(rndp,:);
for kk = 1:(K-1)
    x(:,:,kk) = X(((kk-1)*num+1):(kk*num),:);
    y(:,:,kk) = Y(((kk-1)*num+1):(kk*num),:);
    % 随机等分K类，最小二乘法初始化K个回归系数
    B(:,kk) = (x(:,:,kk)'*x(:,:,kk))\(x(:,:,kk)'*y(:,:,kk));
    Xk = Y - X*B(:,kk);
    Sigma(kk,1) = (Xk'*Xk)/num;
end
% where kk == K 
xx(:,:,K) = X(((K-1)*num+1):end,:);
yy(:,:,K) = Y(((K-1)*num+1):end,:);
B(:,K) = (xx(:,:,K)'*xx(:,:,K))\(xx(:,:,K)'*yy(:,:,K));
Xk = Y - X*B(:,K);
Sigma(K,1) = (Xk'*Xk)/(num+rnum);

pGamma = 1/K*ones(N,K);
end

%% EM更新求解超平面的概率时
function [Px,total_residule] = calc_prob()
Px = zeros(N, K);
% total_residule: N*1, 第i行存储的是第i个样本到所有平面的残差的绝对值的和
residul = zeros(K,1);
for k = 1:K
    Bk = B(:,k);
    pGamma_k = diag(pGamma(:,k));
    residul(k) = (Y-X*Bk)'*pGamma_k*(Y-X*Bk);     
end
% Px:N*K, 位置(i,j)指第i个样本在第j个超平面的残差比例
for kk = 1:K
    Xt = Y - X*B(:,kk);
    tmp = (Xt.*Xt)/Sigma(kk);
    coef = (2*pi)^(-1/2)/sqrt(Sigma(kk));
    Px(:, kk) = coef * exp(-0.5*tmp);
end
total_residule = sum(residul);  % 计算所有样本到所有平面的残差和
end


end


