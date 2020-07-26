function varargout = FCRM_Alpha(data,K,m,Alpha,B_initial)
%   {x_i,y_i} from K hyperplanes with parametres B1,B2，...,B_k 
%   i = 1,2,...(n1+n2)
%   K个超平面
%   m :fuzzy c-regression model的参数
%   Alpha:FCRM_Alpha算法中参数Alpha，更小的Alpha对异常值或噪声更不敏感，
%         Alpha =1时与FCRM相同；
%         when K=2,Alpha \in [0.65,0.6];
%         K=3,Alpha \in [0.6,0.55];
%         K>3,Alpha \in [0.5,0.55];
%   B_initial： K个初始的超平面系数，d*K
%   输出cluster_output：结构体

X = data(:,1:(end-1));
Y = data(:,end);

% 停机误差限
threshold = 1e-6; 

% 获取样本量N，样本维度D
[N, D] = size(X);
%% initial values for hyperplanes' parametres
if nargin > 4
    B = B_initial;    
    e_ik = error_ik(B_initial);
    U_ik = fuzzy_u_ik(e_ik);
else
    B = init_params();
    B_initial = B;
    e_ik = error_ik(B_initial);
    U_ik = fuzzy_u_ik(e_ik);
end
%% 停机方式选择，目标函数第j+1次迭代值与第j次迭代目标值的差异,设迭代步数不超过10000
Obj_Iter_Values = zeros(1000,1);  % 预分配内存，存储每步迭代的目标函数值

t = tic;
j = 1 ;  % 迭代次数，不超过10000

while j<1000
    total_residule = calcu_residule(B,U_ik);
    Obj_Iter_Values(j+1) = total_residule;
    pW = U_ik.^m;

    % new value for parameters of each Component
    Nk = sum(U_ik, 1);   % pW 按列求和，得1*K的矩阵
    pPi = Nk/N;            %这里的pPi也是（1，K）大小,是各簇中样本占总样本量的概率
    shift_X = zeros(N,D,K);
    shift_Y = zeros(N,1,K);
    % 加权最小二乘计算回归系数
    for k =1:K
    shift_X(:,:,k) = X.*repmat(pW(:,k),1,D);
    shift_Y(:,:,k) = Y.*repmat(pW(:,k),1,1);
    %！！！！在迭代更新回归系数beta时，出现线性方程系数矩阵奇异怎么办！！！
    B(:,k) = (shift_X(:,:,k)'*shift_X(:,:,k))\(shift_X(:,:,k)'*shift_Y(:,:,k));
    %！！！！在迭代更新回归系数beta时，出现线性方程系数矩阵奇异怎么办！！！
    end
    % check for convergence 
    if abs(Obj_Iter_Values(j+1) - Obj_Iter_Values(j)) < (threshold)
        disp(['第',num2str(j),'次迭代的总残差前后两步差的绝对值小于1e-6']);
        break;
    end
    %   if code run more than 10s ,stop it whether or not convergent
    if toc(t) > 50
        break;
    end 
    j = j + 1;
    e_ik = error_ik(B);
    U_ik = fuzzy_u_ik(e_ik);
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
function B = init_params()    

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
end
% where kk == K 
xx(:,:,K) = X(((K-1)*num+1):end,:);
yy(:,:,K) = Y(((K-1)*num+1):end,:);
B(:,K) = (xx(:,:,K)'*xx(:,:,K))\(xx(:,:,K)'*yy(:,:,K));

end


%% 计算模糊矩阵U_ik
function U_ik = fuzzy_u_ik(e_ik)
% 输入计算好的 ：e_ik ，第i样本到第k超平面的误差平方
% U_ik ：模糊矩阵U_ik ,N*K

U_ik = zeros(N,K);
m_1 = 1/(m - 1);
% 按行统计e_ki，每行0元素的个数 e_ki_row_0
e_ki_row_0 = sum(e_ik==0, 2);
not_0_id = find(e_ki_row_0~=0);

if ~isempty(not_0_id)
    Ii_ = zeros(N,1);
    Ii_(not_0_id,:) = 1./e_ki_row_0(not_0_id);  
    U_ik = (e_ik==0).*repmat(Ii_,1,K);
end
%e_ki每行,没有0元素的索引id_0
id_0 = find(e_ki_row_0==0);
n0 = length(id_0);
for i = 1:n0
    ei = e_ik(id_0(i),:);
    for k = 1:K
        eci_eli = (ei(k)./ei).^m_1;
        U_ik(id_0(i),k) = 1/sum(eci_eli);
    end
end
[U_row_max,max_id] = max(U_ik,[],2);
id_update = find(U_row_max>Alpha);
% 可知，当Alpha=1时，U_ik不更新
nl =length(id_update);
%% 对超过Alpha的Uik,更新
if nl>0
    for s = 1:nl
%         U_ik(id_update(s),:) = (U_ik(id_update(s),:)==U_row_max(id_update(s)));
        U_ik(id_update(s),:) = 0;
        U_ik(id_update(s),max_id(id_update(s))) = 1;       
    end
end

end
%% 计算e_ik矩阵，第i样本到簇k的误差平方
function e_ik = error_ik(Beta)
% Beta: K个超平面系数，d*K
% e_ki:第i样本到簇k的误差平方,N*K
% 
Y_K = repmat(Y, 1, K); 
XB = X*Beta;
gap = Y_K - XB;
e_ik = gap.*gap; % 第i样本到簇k的误差平方

end

function total_residule = calcu_residule(B,U_ik)
% 计算总的残差平方和
resid = zeros(K,1);
for jj = 1:K
    Bk = B(:,jj);
    pW_k = diag(U_ik(:,jj)).^m;
    %%%%%%%%%%%%%%%%%%%%%%%%%%% 可将pGamma_k先不加权计算，看看结果
    resid(jj) = (Y-X*Bk)'*pW_k*(Y-X*Bk);     
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
end
total_residule = sum(resid);  % 计算所有样本到所有平面的残差和
end

end