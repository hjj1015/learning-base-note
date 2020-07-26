function cluster_output = RL_FCRM(data,stop_epsilon,initial)
% �İ�RL_FCM:  "Robust-learning fuzzy c-means clustering algorithm with
% unknown number of clusters" ,2017 May

% data : n*(d+1)����
% stop_epsilon:ͣ�������
% initial:������ʼ����initial��ʽΪ�ṹ��
x = data(:,1:(end-1));
y = data(:,end);
[n,d] = size(x);
if nargin>2
    % ��ʼ������
    c = initial.c;
    B = initial.B; % d*c ,c����ʼ�ع�ϵ��
    alpha = 1/c*ones(1,c);  % ��ʼ��ÿ���صĻ�ϱ���
    r1 = 1; % ģ�Ͳ���
    r2 = 1; % ģ�Ͳ���
    r3 = 1; % ģ�Ͳ���
else
%     initial = initial_parameter(x,y);
    initial = initial_parameter2(x,y);
    c = initial.c;
    B = initial.B; % d*c ,c����ʼ�ع�ϵ��
    alpha = 1/c*ones(1,c);  % ��ʼ��ÿ���صĻ�ϱ���, c*1
    r1 = 1; % ģ�Ͳ���
    r2 = 1; % ģ�Ͳ���
    r3 = 1; % ģ�Ͳ���
end
% Ԥ�����ڴ�
t_max = 200; % ��������������Ϊ200
eta = min(1,2/(d^fix(d/2-1))); % ����r3�Ķ������
c_hist = zeros(t_max,1);   % ��¼ÿ�ε�������Ŀ�仯

r1_hist = zeros(t_max,1);  % ��¼ÿ�ε������ڲ���r1�ı仯
r2_hist = zeros(t_max,1);  % ��¼ÿ�ε������ڲ���r2�ı仯
r3_hist = zeros(t_max,1);  % ��¼ÿ�ε������ڲ���r3�ı仯

% ��������t
t = 1;
% compute:sample membership value of cluster
Uik = Compute_Uik(B,alpha,c,r1,r2);

% ������һ�ε�c,B,alpha,r1,r2,r3
c_hist(t) = c;
B_hist= B;          % ������һ�������Ļع�ϵ��
alpha_hist = alpha; % ������һ�������Ļ�ϱ���alpha
r1_hist(t) = r1; 
r2_hist(t) = r2; 
r3_hist(t) = r3;

% ����r1 ,r2
r1 = exp(-t/10);
r2 = exp(-t/100);
% ���»�ϱ���alpha��������һ���Ļ�ϱ���alpha_hist��Uik
alpha = update_alpha(Uik,alpha_hist,r1,r3);
% ����r3
r3 = update_r3(alpha_hist,alpha,Uik);
% �����µĴ���Ŀc,������������Ĵ���
[c,inx] = update_c(c_hist(t),alpha);

% ����������������󣬱�׼��alpha��Uik
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
    % ������һ�ε�c,B,alpha,r1,r2,r3
    c_hist(t) = c;
    B_hist= B;          % ������һ�������Ļع�ϵ��
    alpha_hist = alpha; % ������һ�������Ļ�ϱ���alpha
    r1_hist(t) = r1;
    r2_hist(t) = r2;
    r3_hist(t) = r3;
    
    % ����r1 ,r2
    r1 = exp(-t/10);
    r2 = exp(-t/100);
    % ���»�ϱ���alpha��������һ���Ļ�ϱ���alpha_hist��Uik
    alpha = update_alpha(Uik,alpha_hist,r1,r3);  %#############  ���µ�alpha��Ȼ�и�ֵ������
    % ����r3
    r3 = update_r3(alpha_hist,alpha,Uik);
    % ���´���Ŀc,������������Ĵ���
    [c,inx] = update_c(c_hist(t),alpha);         %#############�����¸��´�cҲ����������

    if c<2
        c = c_hist(t);
        alpha = alpha_hist;
        break
    end
    % ����������������󣬱�׼��alpha��Uik
    alpha = normlize_alpha(alpha,inx);
    Uik = normlize_Uik(Uik,inx);

    if (t>100) && (c_hist(t-100)==c_hist(t))
        r3 = 0;
    end
    B = update_B(Uik,c);
    dis_B = Gap_B_iter(B,B_hist,c);

end
% ������

cluster_output.t = t;
cluster_output.Uik = Uik;
cluster_output.alpha = alpha;
cluster_output.B = B;
cluster_output.C = c;
cluster_output.c_hist = c_hist(1:t,1);
r = [r1_hist,r2_hist,r3_hist];
cluster_output.r= r(1:t,:);





    function initial = initial_parameter(x,y)
        % ��ʼ����c��c��dά�ع�ϵ������B \in R^{d*c}
        [n,d] = size(x);
        % ����c���أ�ÿ���طֵ�������ʼԼΪn/d������Ϊ����dά�ع�ϵ��������Ҫd������
        % ����Ҳ�ɵ���dΪ(d+1)��������Щ
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
        % ��ʼ����c��c��dά�ع�ϵ������B \in R^{d*c}
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
        % B:d*c���ع�ϵ������
        % alpha:1*c,�����ػ�ϱ���
        % Uik:n*c��ģ��ֵ����
        
        Alpha_mat = repmat(alpha,n,1);  % n*c
        dis_ik = dis_to_plane(B);  % n*c
        inter_calcu = r1*log(Alpha_mat) - dis_ik ; 
        inter_calcu = inter_calcu/r2;
        inter_calcu = exp(inter_calcu);
        row_sum = sum(inter_calcu,2); % �����к�,n*1;
        Uik = inter_calcu./repmat(row_sum,1,c);
    end

    function alpha_new = update_alpha(Uik,alpha_old,r1,r3)
        % Uik:n*c��ģ��ֵ����
        % alpha_hist: 1*c
        column_sum_Uik = sum(Uik,1);  % �����У�1*c
        ln_alpha_old = log(alpha_old);  % 1*c
        alpha_LnAlpha = alpha_old.*ln_alpha_old; % 1*c
        sum2 = sum(alpha_LnAlpha);  % 1��c�����
        inter_calcu = alpha_LnAlpha - alpha_old*sum2;
        alpha_new = column_sum_Uik/n + r3/r1*inter_calcu;
    end

    function r3 = update_r3(alpha_old,alpha_new,Uik)
        % ��alpha_old��alpha_new����r3
        value1 = -eta*n*abs(alpha_new-alpha_old);
        value1 = exp(value1);
        value1 = mean(value1);
        mean_Uik = mean(Uik);
        value2 = max(mean_Uik) - 1;
        inter_calcu = sum(alpha_old.*log(alpha_old));  % һ��ʵ��
        value2 = value2/max(alpha_old*inter_calcu);
        r3 = min(value1,value2);
    end

    function [c,inx] = update_c(c_hist,alpha_new)
        % ���´���Ŀ��������һЩ������Ĵ���
        % c_hist����һ�������Ĵ���
        % ���������µ�alphak������: inx
        
        % ���㲻�����alpha_new������λ��Alpha_less_1_n
        Alpha_less_1_n = find(alpha_new<(1/n));
        % ������Ĵ���Ŀ
        delete_cluster_n = length(Alpha_less_1_n);
        c = c_hist - delete_cluster_n;
        inx = setdiff(1:c_hist,Alpha_less_1_n);
    end

    function     alpha = normlize_alpha(alpha_old,inx)
        % ����������Ĵغ����׼��alpha
        % ���뱣���µ�alpha��Ӧ������λinx
        alpha_new = alpha_old(inx);
        alpha = alpha_new/sum(alpha_new);       
    end

    function Uik = normlize_Uik(Uik_old,inx)
        % ����������Ĵغ����׼��Uik
        % ���뱣���µ�alpha��Ӧ������λinx
        c_new = length(inx);
        Uik_new = Uik_old(:,inx);

        Uik = Uik_new./repmat(sum(Uik_new,2),1,c_new);
        
    end

    function B = update_B(Uik,c)
        % ���»ع�ϵ��B��Uik�Ƕ������Ϸ��Ĵ����ٱ�׼�����ģ��ֵ����     
        % B: d*c
            shift_X = zeros(n,d,c);
            shift_Y = zeros(n,1,c);
            % ��Ȩ��С���˼���ع�ϵ��
            for k =1:c
                shift_X(:,:,k) = x.*repmat(Uik(:,k),1,d);
                shift_Y(:,:,k) = y.*repmat(Uik(:,k),1,1);
                %�����������������������ܳ���ϵ���������죡����������������������������
                A = shift_X(:,:,k)'*shift_X(:,:,k);
                B(:,k) = A\(shift_X(:,:,k)'*shift_Y(:,:,k)); % �������� ��������������
                %�����������������������ܳ���ϵ���������죡����������������������������
            end
    end

    function dis_B = Gap_B_iter(B,B_hist,c)
        % ǰ�����Bk�ع�ϵ��c_old���е�������
        % B: ��ǰ���ع�ϵ������d*c_new
        % B_hist�� ǰһ���ع�ϵ������d*c_old
        % c_new<=c_old
        B1 = B_hist(:,1:c);
        B_iter_Gap = B1 - B;
        % ����B_iter_Gap�з�����ÿ�о���ֵ�͵����ֵ
        dis_B = norm(B_iter_Gap,1);
    end

    function dis_ik = dis_to_plane(B)
        % ����n�������㵽�ع�ϵ����Ӧ��ƽ�����
        % B:d*c
        % dis_ik: n*c��������ƽ��
        y_xb = repmat(y,1,c) - x*B;
        dis_ik = y_xb.*y_xb;
    end
       

end

