function cluster_output = Robust_spasity_outlier_FCRM(data,K,m,lambda,initial)
% Reference:Forero P A , Kekatos V , Giannakis G B. 
% Robust Clustering Using Outlier-Sparsity Regularization[J]. IEEE Transactions on Signal Processing, 2012, 60(8):4163-4177.
% data : n*(d+1)����
% stop_epsilon:ͣ�������
% C:�ع鳬ƽ�����
% m:ģ�����ӣ�m=1.5(������)
% lambda : ģ�ͳ�����������ϡ����
% initial:������ʼ����initial��ʽΪ�ṹ��

% cluster_output��������

x = data(:,1:(end-1));
y = data(:,end);
[N,D] = size(x);
stop_epsilon = 1e-6;
Outlier = zeros(N,1); % ��ʼ��n��������Я�����쳣ֵ��С��Ϊ0
gamma = 1.7;  % MCP����

if nargin>4
    % ��ʼ������
    B_initial = initial.B; % D*K ,K����ʼ�ع�ϵ��  
else
    B_initial = initial_parameter(x,y);
end
% ���ݳ�ʼ����B������ʼ��U����
Uik = Compute_Uik(x,y,B_initial,Outlier,lambda);

t_max = 1000; % ����������
% Ԥ�����ڴ�Outlier_hist
Outlier_hist = zeros(N,t_max);
% ������һ����ģ������Uik
Uik_hist = Uik;  
for t=1:t_max
    % ���»ع�ϵ������
    B = update_B(x,y,Uik_hist,Outlier,m);
    % �����쳣ֵ����,��Ҫϡ����⣬��MCP��
    Outlier = update_Outlier(x,y,Uik_hist,B,lambda);
    % ���浱ǰ�����쳣ֵ����
    Outlier_hist(:,t) = Outlier;
    % ����ģ������Uik
    Uik = Compute_Uik(x,y,B,Outlier,lambda);  
    % �ж�ͣ��׼��
    Uik_nm = norm(Uik,'fro');
    Uik_Uik_hist = Uik - Uik_hist;
    Uik_Uik_hist_gap = norm(Uik_Uik_hist,'fro');
    
    if Uik_Uik_hist_gap/Uik_nm<=stop_epsilon
        break
    end
end
% ���������
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
            % ����ȷ�K�࣬��С���˷���ʼ��K���ع�ϵ��
            B(:,kk) = (x_(:,:,kk)'*x_(:,:,kk))\(x_(:,:,kk)'*y_(:,:,kk));
        end
        % where kk == K
        xx(:,:,K) = X(((K-1)*num+1):end,:);
        yy(:,:,K) = Y(((K-1)*num+1):end,:);
        B(:,K) = (xx(:,:,K)'*xx(:,:,K))\(xx(:,:,K)'*yy(:,:,K)); 
    end

    %% ����ģ������U_ik   !!!!!!!!!!!
    function U_ik = Compute_Uik(x,y,B,Outlier,lambda) 
        % ���� ��B ���ع�ϵ������D*K��
        % Outlier: N*1 ,N���쳣ֵ����
        % U_ik ��ģ������U_ik ,N*K
        U_ik = zeros(N,K);
        
        m_1 = 1/(m - 1);
        % ����ͳ��e_ki��ÿ��0Ԫ�صĸ��� e_ki_row_0
        e_ik = error_ik(x,y,B,Outlier,lambda);

        e_ki_row_0 = sum(e_ik==0, 2);
        not_0_id = find(e_ki_row_0~=0);
        
        if ~isempty(not_0_id)
            Ii_ = zeros(N,1);
            Ii_(not_0_id,:) = 1./e_ki_row_0(not_0_id);
            U_ik = (e_ik==0).*repmat(Ii_,1,K);
        end
        %e_kiÿ��,û��0Ԫ�ص�����id_0
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

    %% ����e_ik���󣬵�i��������k�����ƽ��
    function e_ik = error_ik(X,Y,Beta,Outlier,lambda)
        % Beta: K����ƽ��ϵ����D*K
        % e_ki:��i��������k�����ƽ��,N*K
        %
        Y_K = repmat(Y, 1, K);  % N*K
        Outlier_K = repmat(Outlier, 1, K);  % N*K
        XB = X*Beta;      % N*K 
        gap = Y_K - XB - Outlier_K;
        e_ik = gap.*gap; % ��i��������k�����ƽ��   
        lamda_On_nm = lambda*abs(Outlier_K);
        e_ik = e_ik + lamda_On_nm;
    end
    
    function B = update_B(x,y,Uik,Outlier,m) 
    % ���»ع�ϵ��B��Uik��ģ��ֵ����
    % B: D*K
    B = zeros(D,K);
    shift_X = zeros(N,D,K);
    shift_Y = zeros(N,1,K);
    % ��Ȩ��С���˼���ع�ϵ��
    y = y - Outlier;
    pW = Uik.^m;
    for c =1:K
        shift_X(:,:,c) = x.*repmat(pW(:,c),1,D);
        shift_Y(:,:,c) = y.*repmat(pW(:,c),1,1);
        %�����������������������ܳ���ϵ���������죡����������������������������
        A = shift_X(:,:,c)'*shift_X(:,:,c);
        B(:,c) = A\(shift_X(:,:,c)'*shift_Y(:,:,c)); % �������� ��������������
        %�����������������������ܳ���ϵ���������죡����������������������������
    end
    
    end

    function Outlier = update_Outlier(x,y,Uik_hist,B,lambda)
    % input: ��һ����Uik,��ǰ����B�������쳣ֵ����
    % Outlier�� N*1
    % METHOD: ϡ����⣬��MCP��;code�ȳ��Ե��������еĵ�����ʽ
    % METHOD: MCP
    Uik_m = Uik_hist.^m;
    dis = repmat(y,1,K) - x*B;  % N*K
    dis = Uik_m.*dis;
    Outlier = zeros(N,1);

    dis_sum_row = sum(dis,2);    % ���������,N*1
    sum_Uik_m_row = sum(Uik_m,2);% ���������,N*1
    R = dis_sum_row./sum_Uik_m_row; % N*1,����N��������Ӧ��r_n
    
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