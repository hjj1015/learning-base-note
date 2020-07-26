function varargout = EM_for_hyperplane(X,Y,K,B_initial)
%   X :(n1+n2)*dim
%   Y:(n1+n2)*1
%   {x_i,y_i} from two hyperplanes with parametres B1,B2 ,
%   {x_i,y_i} subject to y_i = x_i*B1 OR y_i = x_i*B2,
%   i = 1,2,...(n1+n2)

%  当多类别时，如何根据残差更新样本i到类别k的概率pk
% [di_sort,ix_] = sort(di)  %di已经被单位化了，di = di/sum(di)
% cluser_num = length(di)
% [~,ix] = sort(ix_);
% pk = di_sort(cluser_num+1-ix)

threshold = 1e-15;
[N, D] = size(X);

% initial values for hyperplanes' parametres
if nargin > 3
    B = B_initial;
    pPi = 1/K*ones(1,K);
else
    [B,pPi] = init_params();
end

%  B_1_before for saving B1 of previous step calculating in oder to judge
%  whether code can stop.
B_1_before = zeros(D,1);
total_residule_before = 0;

t = tic;
j = 1 ;
while true
    [Px,total_residule] = calc_prob();
    pGamma = Px .* repmat(pPi, N, 1);  % pGamma：N*K
    pGamma = pGamma ./ repmat(sum(pGamma, 2), 1, K);

    % new value for parameters of each Component
    Nk = sum(pGamma, 1);   % pGamma 按列求和，得1*K的矩阵
    pPi = Nk/N;    %这里的pPi也是（1，K）大小,是各簇中样本占总样本量的概率
    shift_X = zeros(N,D,K);
    shift_Y = zeros(N,1,K);
    for k =1:K
    shift_X(:,:,k) = X.*repmat(Px(:,k),1,D)/Nk(k); % ！！！！！！！！
    shift_Y(:,:,k) = Y.*repmat(Px(:,k),1,1)/Nk(k); % ！！！！！！！！
    B(:,k) = (shift_X(:,:,k)'*shift_X(:,:,k))\(shift_X(:,:,k)'*shift_Y(:,:,k));
    end
    % check for convergence  ???这个停机准则是不是太苛刻，一般都达不到这条准则，修改！！用残差停，残差不再减小
    B_1 = B(:,1);
    if norm(B_1 -B_1_before) < threshold
        disp(['第',num2str(j),'次迭代前后两步系数B1的差距二范数小于1e-15']);
        disp(['平均每个维度上的前后差距为:',num2str(norm(B_1 -B_1_before)/dim)]);
        break;
    end
    if sum(abs(total_residule - total_residule_before)) < (threshold*1000)
        disp(['第',num2str(j),'次迭代的总残差前后两步差的绝对值小于1e-12']);
        break;
    end
    total_residule_before = total_residule ;
    B_1_before = B_1;
    %   if code run more than 10s ,stop it whether or not convergent
    if toc(t) > 50
        break;
    end 
    j = j + 1;
%     toc(t);
end
if nargout == 1
    varargout = {Px};
else
    model = [];
    model.B = B;
%     model.B1 = B(:,1);
%     model.B2 = B(:,2);
%     [x11,y11,x22,y22] = USE_coeffi_TO_CLASSIFY_20(X,Y,B(:,1),B(:,2));
%     model.B1 = (x11'*x11)\(x11'*y11);
%     model.B2 = (x22'*x22)\(x22'*y22);
    model.pPi = pPi;
    varargout = {Px, model};
    disp('total_residule');
    disp(total_residule);
    disp('total_av_residule');
    disp(total_residule/N);
end

function [B,pPi] = init_params()
pPi = 1/K*ones(1,K);     %K表示有K个超平面
num = N/K;
x = zeros(num,D,K);
y = zeros(num,1,K);
B = zeros(D,K);
rndp = randperm(N);
X = X(rndp,:);
Y = Y(rndp,:);
for kk = 1:K
    x(:,:,kk) = X(((kk-1)*num+1):(kk*num),:);
    y(:,:,kk) = Y(((kk-1)*num+1):(kk*num),:);
    B(:,kk) = (x(:,:,kk)'*x(:,:,kk))\(x(:,:,kk)'*y(:,:,kk));
end
end

function [Px,total_residule] = calc_prob()
Px = zeros(N, K);
total_residule = sum(abs(repmat(Y,1,K)-X*B),2); % total_residule:N*1,第i行存储的是第i个样本到所有平面的残差的绝对值的和
% for i = 1:K
%     Px(:,i) = abs(Y-X*B(:,i))./total_residule;
% end
Px = abs(repmat(Y,1,K)-X*B)./(repmat(total_residule,1,K));%Px:N*K, 位置(i,j)指第i个样本在第j个超平面的残差比例
%% 将距离绝对值偏差大的，赋以相反的概率
for jj = 1:N
    [di_sort,ix_] = sort(Px(jj,:));    
    [~,ix] = sort(ix_);
    Px(jj,:) = di_sort(K+1-ix);
end
% Px = ones(N, K)-Px;


total_residule = sum(total_residule);  % 计算所有样本到所有平面的残差和
end

end





