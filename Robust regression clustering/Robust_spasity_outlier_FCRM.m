function cluster_output = Robust_spasity_outlier_FCRM(data,K,m,lambda,initial)
% Reference:Forero P A , Kekatos V , Giannakis G B. 
% Robust Clustering Using Outlier-Sparsity Regularization[J]. IEEE Transactions on Signal Processing, 2012, 60(8):4163-4177.
% data : n*(d+1)样本
% stop_epsilon:停机误差限
% C:回归超平面个数
% m:模糊因子，m=1.5(该文献)
% lambda : 模型超参数，控制稀疏性
% initial:参数初始化，initial格式为结构体

% cluster_output：聚类结果

x = data(:,1:(end-1));
y = data(:,end);
[N,D] = size(x);
stop_epsilon = 1e-6;
Outlier = zeros(N,1); % 初始化n个样本，携带的异常值大小均为0
gamma = 1.7;  % MCP参数

if nargin>4
    % 初始化参数
    B_initial = initial.B; % D*K ,K个初始回归系数  
else
    B_initial = initial_parameter(x,y);
end
% 依据初始化的B，来初始化U矩阵
Uik = Compute_Uik(x,y,B_initial,Outlier,lambda);

t_max = 1000; % 最大迭代次数
% 预分配内存Outlier_hist
Outlier_hist = zeros(N,t_max);
% 保存上一步的模糊矩阵Uik
Uik_hist = Uik;  
for t=1:t_max
    % 更新回归系数矩阵
    B = update_B(x,y,Uik_hist,Outlier,m);
    % 更新异常值能量,需要稀疏求解，如MCP法
    Outlier = update_Outlier(x,y,Uik_hist,B,lambda);
    % 保存当前步的异常值能量
    Outlier_hist(:,t) = Outlier;
    % 更新模糊矩阵Uik
    Uik = Compute_Uik(x,y,B,Outlier,lambda);  
    % 判定停机准则
    Uik_nm = norm(Uik,'fro');
    Uik_Uik_hist = Uik - Uik_hist;
    Uik_Uik_hist_gap = norm(Uik_Uik_hist,'fro');
    
    if Uik_Uik_hist_gap/Uik_nm<=stop_epsilon
        break
    end
end
% 输出聚类结果
Outlier_hist = Outlier_hist(:,1:t);
cluster_output.Outlier_hist = Outlier_hist;
cluster_output.Uik = Uik;
cluster_output.B = B;

    function B = initial_parameter(X,Y)
        
        num = fix(N/K);
        rnum = N-num*K;
        
        x_ = zeros(num,D,K);
        y_ = zeros(num,1,K);
        xx = zeros(num+rnum,D,K);
        yy = zeros(num+rnum,1,K);
        
        B = zeros(D,K);
        rndp = randperm(N);
        X = X(rndp,:);
        Y = Y(rndp,:);
        for kk = 1:(K-1)
            x_(:,:,kk) = X(((kk-1)*num+1):(kk*num),:);
            y_(:,:,kk) = Y(((kk-1)*num+1):(kk*num),:);
            % 随机等分K类，最小二乘法初始化K个回归系数
            B(:,kk) = (x_(:,:,kk)'*x_(:,:,kk))\(x_(:,:,kk)'*y_(:,:,kk));
        end
        % where kk == K
        xx(:,:,K) = X(((K-1)*num+1):end,:);
        yy(:,:,K) = Y(((K-1)*num+1):end,:);
        B(:,K) = (xx(:,:,K)'*xx(:,:,K))\(xx(:,:,K)'*yy(:,:,K)); 
    end

    %% 计算模糊矩阵U_ik   !!!!!!!!!!!
    function U_ik = Compute_Uik(x,y,B,Outlier,lambda) 
        % 输入 ：B ，回归系数矩阵（D*K）
        % Outlier: N*1 ,N个异常值能量
        % U_ik ：模糊矩阵U_ik ,N*K
        U_ik = zeros(N,K);
        
        m_1 = 1/(m - 1);
        % 按行统计e_ki，每行0元素的个数 e_ki_row_0
        e_ik = error_ik(x,y,B,Outlier,lambda);

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
        
    end

    %% 计算e_ik矩阵，第i样本到簇k的误差平方
    function e_ik = error_ik(X,Y,Beta,Outlier,lambda)
        % Beta: K个超平面系数，D*K
        % e_ki:第i样本到簇k的误差平方,N*K
        %
        Y_K = repmat(Y, 1, K);  % N*K
        Outlier_K = repmat(Outlier, 1, K);  % N*K
        XB = X*Beta;      % N*K 
        gap = Y_K - XB - Outlier_K;
        e_ik = gap.*gap; % 第i样本到簇k的误差平方   
        lamda_On_nm = lambda*abs(Outlier_K);
        e_ik = e_ik + lamda_On_nm;
    end
    
    function B = update_B(x,y,Uik,Outlier,m) 
    % 更新回归系数B，Uik是模糊值矩阵
    % B: D*K
    B = zeros(D,K);
    shift_X = zeros(N,D,K);
    shift_Y = zeros(N,1,K);
    % 加权最小二乘计算回归系数
    y = y - Outlier;
    pW = Uik.^m;
    for c =1:K
        shift_X(:,:,c) = x.*repmat(pW(:,c),1,D);
        shift_Y(:,:,c) = y.*repmat(pW(:,c),1,1);
        %！！！！！！！！！！可能出现系数矩阵奇异！！！！！！！！！！！！！！！
        A = shift_X(:,:,c)'*shift_X(:,:,c);
        B(:,c) = A\(shift_X(:,:,c)'*shift_Y(:,:,c)); % 出现奇异 ！！！！！！！
        %！！！！！！！！！！可能出现系数矩阵奇异！！！！！！！！！！！！！！！
    end
    
    end

    function Outlier = update_Outlier(x,y,Uik_hist,B,lambda)
    % input: 上一步的Uik,当前步的B，更新异常值能量
    % Outlier： N*1
    % METHOD: 稀疏求解，如MCP法;code先尝试的是论文中的迭代格式
    % METHOD: MCP
    Uik_m = Uik_hist.^m;
    dis = repmat(y,1,K) - x*B;  % N*K
    dis = Uik_m.*dis;
    Outlier = zeros(N,1);

    dis_sum_row = sum(dis,2);    % 按照行求和,N*1
    sum_Uik_m_row = sum(Uik_m,2);% 按照行求和,N*1
    R = dis_sum_row./sum_Uik_m_row; % N*1,包含N个样本对应的r_n
    
    Rn_nm = abs(R); % N*1
    lm = lambda/2;
    gamma_lamda = gamma*lm;
    gama1 = gamma/(gamma-1);
    
    id_more_rlamda = find(Rn_nm>gamma_lamda);
    id_less_rlamda = setdiff([1:N]',id_more_rlamda);
    
    inter_variable = 1 - lm./Rn_nm;                 % N*1
    positive_operator = max(inter_variable,0);      % N*1
    INTER_VALUE = R.* positive_operator;            % s(theta;lambda)
    Outlier(id_less_rlamda,1) = gama1*INTER_VALUE(id_less_rlamda,1) ;  
    Outlier(id_more_rlamda,1) = R(id_more_rlamda,1) ; 

    end


    end