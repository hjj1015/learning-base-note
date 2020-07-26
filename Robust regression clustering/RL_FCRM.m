function cluster_output = RL_FCRM(data,stop_epsilon,initial)
% 改版RL_FCM:  "Robust-learning fuzzy c-means clustering algorithm with
% unknown number of clusters" ,2017 May

% data : n*(d+1)样本
% stop_epsilon:停机误差限
% initial:参数初始化，initial格式为结构体
x = data(:,1:(end-1));
y = data(:,end);
[n,d] = size(x);
if nargin>2
    % 初始化参数
    c = initial.c;
    B = initial.B; % d*c ,c个初始回归系数
    alpha = 1/c*ones(1,c);  % 初始的每个簇的混合比例
    r1 = 1; % 模型参数
    r2 = 1; % 模型参数
    r3 = 1; % 模型参数
else
%     initial = initial_parameter(x,y);
    initial = initial_parameter2(x,y);
    c = initial.c;
    B = initial.B; % d*c ,c个初始回归系数
    alpha = 1/c*ones(1,c);  % 初始的每个簇的混合比例, c*1
    r1 = 1; % 模型参数
    r2 = 1; % 模型参数
    r3 = 1; % 模型参数
end
% 预分配内存
t_max = 200; % 假设最大迭代次数为200
eta = min(1,2/(d^fix(d/2-1))); % 调节r3的额外参数
c_hist = zeros(t_max,1);   % 记录每次迭代簇数目变化

r1_hist = zeros(t_max,1);  % 记录每次迭代调节参数r1的变化
r2_hist = zeros(t_max,1);  % 记录每次迭代调节参数r2的变化
r3_hist = zeros(t_max,1);  % 记录每次迭代调节参数r3的变化

% 迭代次数t
t = 1;
% compute:sample membership value of cluster
Uik = Compute_Uik(B,alpha,c,r1,r2);

% 储存上一次的c,B,alpha,r1,r2,r3
c_hist(t) = c;
B_hist= B;          % 保存上一步迭代的回归系数
alpha_hist = alpha; % 保存上一步迭代的混合比例alpha
r1_hist(t) = r1; 
r2_hist(t) = r2; 
r3_hist(t) = r3;

% 更新r1 ,r2
r1 = exp(-t/10);
r2 = exp(-t/100);
% 更新混合比例alpha，输入上一步的混合比例alpha_hist和Uik
alpha = update_alpha(Uik,alpha_hist,r1,r3);
% 更新r3
r3 = update_r3(alpha_hist,alpha,Uik);
% 更新新的簇数目c,丢弃到不合理的簇数
[c,inx] = update_c(c_hist(t),alpha);

% 丢弃掉不合理簇数后，标准化alpha和Uik
alpha = normlize_alpha(alpha,inx);

Uik = normlize_Uik(Uik,inx);

if (t>100) && (c_hist(t-100)==c_hist(t))
    r3 = 0;
end
B = update_B(Uik,c);
dis_B = Gap_B_iter(B,B_hist,c);

while (dis_B>=stop_epsilon) && (t<200)
    t = t + 1;
    % compute:sample membership value of cluster
    Uik = Compute_Uik(B,alpha,c,r1,r2);
    % 储存上一次的c,B,alpha,r1,r2,r3
    c_hist(t) = c;
    B_hist= B;          % 保存上一步迭代的回归系数
    alpha_hist = alpha; % 保存上一步迭代的混合比例alpha
    r1_hist(t) = r1;
    r2_hist(t) = r2;
    r3_hist(t) = r3;
    
    % 更新r1 ,r2
    r1 = exp(-t/10);
    r2 = exp(-t/100);
    % 更新混合比例alpha，输入上一步的混合比例alpha_hist和Uik
    alpha = update_alpha(Uik,alpha_hist,r1,r3);  %#############  更新的alpha居然有负值，导致
    % 更新r3
    r3 = update_r3(alpha_hist,alpha,Uik);
    % 更新簇数目c,丢弃到不合理的簇数
    [c,inx] = update_c(c_hist(t),alpha);         %#############，导致更新簇c也出现了问题

    if c<2
        c = c_hist(t);
        alpha = alpha_hist;
        break
    end
    % 丢弃掉不合理簇数后，标准化alpha和Uik
    alpha = normlize_alpha(alpha,inx);
    Uik = normlize_Uik(Uik,inx);

    if (t>100) && (c_hist(t-100)==c_hist(t))
        r3 = 0;
    end
    B = update_B(Uik,c);
    dis_B = Gap_B_iter(B,B_hist,c);

end
% 输出结果

cluster_output.t = t;
cluster_output.Uik = Uik;
cluster_output.alpha = alpha;
cluster_output.B = B;
cluster_output.C = c;
cluster_output.c_hist = c_hist(1:t,1);
r = [r1_hist,r2_hist,r3_hist];
cluster_output.r= r(1:t,:);





    function initial = initial_parameter(x,y)
        % 初始化簇c和c个d维回归系数矩阵B \in R^{d*c}
        [n,d] = size(x);
        % 假设c个簇，每个簇分的样本初始约为n/d个，因为估计d维回归系数至少需要d个样本
        % 这里也可调整d为(d+1)甚至更大些
        c0 = fix(n/d); 
%         B0 = zeros(d*c0);
%         for i = 1:c0
%             B0(:,i) = solve_B(X,Y)
%         end        
        [~, model0] = FCRM_solve_K_plane(x,y,c0,2);
        B0 = model0.B;
        initial.c = c0;
        initial.B = B0;        
    end

    function initial = initial_parameter2(x,y)
        % 初始化簇c和c个d维回归系数矩阵B \in R^{d*c}
        c0 = n; 
        B0 = zeros(d,c0);
        for k = 1:c0
            x_nm = x(k,:)*x(k,:)';
            B0(:,k) = x(k,:)'/x_nm*y(k);
        end
        initial.c = c0;
        initial.B = B0;
    end


    function Uik = Compute_Uik(B,alpha,c,r1,r2)
        % B:d*c，回归系数矩阵
        % alpha:1*c,各个簇混合比例
        % Uik:n*c的模糊值矩阵
        
        Alpha_mat = repmat(alpha,n,1);  % n*c
        dis_ik = dis_to_plane(B);  % n*c
        inter_calcu = r1*log(Alpha_mat) - dis_ik ; 
        inter_calcu = inter_calcu/r2;
        inter_calcu = exp(inter_calcu);
        row_sum = sum(inter_calcu,2); % 计算行和,n*1;
        Uik = inter_calcu./repmat(row_sum,1,c);
    end

    function alpha_new = update_alpha(Uik,alpha_old,r1,r3)
        % Uik:n*c的模糊值矩阵
        % alpha_hist: 1*c
        column_sum_Uik = sum(Uik,1);  % 求列行，1*c
        ln_alpha_old = log(alpha_old);  % 1*c
        alpha_LnAlpha = alpha_old.*ln_alpha_old; % 1*c
        sum2 = sum(alpha_LnAlpha);  % 1到c个求和
        inter_calcu = alpha_LnAlpha - alpha_old*sum2;
        alpha_new = column_sum_Uik/n + r3/r1*inter_calcu;
    end

    function r3 = update_r3(alpha_old,alpha_new,Uik)
        % 用alpha_old，alpha_new更新r3
        value1 = -eta*n*abs(alpha_new-alpha_old);
        value1 = exp(value1);
        value1 = mean(value1);
        mean_Uik = mean(Uik);
        value2 = max(mean_Uik) - 1;
        inter_calcu = sum(alpha_old.*log(alpha_old));  % 一个实数
        value2 = value2/max(alpha_old*inter_calcu);
        r3 = min(value1,value2);
    end

    function [c,inx] = update_c(c_hist,alpha_new)
        % 更新簇数目，丢弃掉一些不合理的簇数
        % c_hist：上一步迭代的簇数
        % 并返回留下的alphak的索引: inx
        
        % 计算不合理的alpha_new的索引位置Alpha_less_1_n
        Alpha_less_1_n = find(alpha_new<(1/n));
        % 不合理的簇数目
        delete_cluster_n = length(Alpha_less_1_n);
        c = c_hist - delete_cluster_n;
        inx = setdiff(1:c_hist,Alpha_less_1_n);
    end

    function     alpha = normlize_alpha(alpha_old,inx)
        % 丢掉不合理的簇后，需标准化alpha
        % 输入保留下的alpha对应的索引位inx
        alpha_new = alpha_old(inx);
        alpha = alpha_new/sum(alpha_new);       
    end

    function Uik = normlize_Uik(Uik_old,inx)
        % 丢掉不合理的簇后，需标准化Uik
        % 输入保留下的alpha对应的索引位inx
        c_new = length(inx);
        Uik_new = Uik_old(:,inx);

        Uik = Uik_new./repmat(sum(Uik_new,2),1,c_new);
        
    end

    function B = update_B(Uik,c)
        % 更新回归系数B，Uik是丢弃不合法的簇数再标准化后的模糊值矩阵     
        % B: d*c
            shift_X = zeros(n,d,c);
            shift_Y = zeros(n,1,c);
            % 加权最小二乘计算回归系数
            for k =1:c
                shift_X(:,:,k) = x.*repmat(Uik(:,k),1,d);
                shift_Y(:,:,k) = y.*repmat(Uik(:,k),1,1);
                %！！！！！！！！！！可能出现系数矩阵奇异！！！！！！！！！！！！！！！
                A = shift_X(:,:,k)'*shift_X(:,:,k);
                B(:,k) = A\(shift_X(:,:,k)'*shift_Y(:,:,k)); % 出现奇异 ！！！！！！！
                %！！！！！！！！！！可能出现系数矩阵奇异！！！！！！！！！！！！！！！
            end
    end

    function dis_B = Gap_B_iter(B,B_hist,c)
        % 前后迭代Bk回归系数c_old个中的最大距离
        % B: 当前步回归系数矩阵，d*c_new
        % B_hist： 前一步回归系数矩阵d*c_old
        % c_new<=c_old
        B1 = B_hist(:,1:c);
        B_iter_Gap = B1 - B;
        % 计算B_iter_Gap列范数，每列绝对值和的最大值
        dis_B = norm(B_iter_Gap,1);
    end

    function dis_ik = dis_to_plane(B)
        % 计算n个样本点到回归系数对应的平面距离
        % B:d*c
        % dis_ik: n*c，二范数平方
        y_xb = repmat(y,1,c) - x*B;
        dis_ik = y_xb.*y_xb;
    end
       

end

