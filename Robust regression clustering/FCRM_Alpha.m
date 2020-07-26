function varargout = FCRM_Alpha(data,K,m,Alpha,B_initial)
%   {x_i,y_i} from K hyperplanes with parametres B1,B2��...,B_k 
%   i = 1,2,...(n1+n2)
%   K����ƽ��
%   m :fuzzy c-regression model�Ĳ���
%   Alpha:FCRM_Alpha�㷨�в���Alpha����С��Alpha���쳣ֵ�������������У�
%         Alpha =1ʱ��FCRM��ͬ��
%         when K=2,Alpha \in [0.65,0.6];
%         K=3,Alpha \in [0.6,0.55];
%         K>3,Alpha \in [0.5,0.55];
%   B_initial�� K����ʼ�ĳ�ƽ��ϵ����d*K
%   ���cluster_output���ṹ��

X = data(:,1:(end-1));
Y = data(:,end);

% ͣ�������
threshold = 1e-6; 

% ��ȡ������N������ά��D
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
%% ͣ����ʽѡ��Ŀ�꺯����j+1�ε���ֵ���j�ε���Ŀ��ֵ�Ĳ���,���������������10000
Obj_Iter_Values = zeros(1000,1);  % Ԥ�����ڴ棬�洢ÿ��������Ŀ�꺯��ֵ

t = tic;
j = 1 ;  % ����������������10000

while j<1000
    total_residule = calcu_residule(B,U_ik);
    Obj_Iter_Values(j+1) = total_residule;
    pW = U_ik.^m;

    % new value for parameters of each Component
    Nk = sum(U_ik, 1);   % pW ������ͣ���1*K�ľ���
    pPi = Nk/N;            %�����pPiҲ�ǣ�1��K����С,�Ǹ���������ռ���������ĸ���
    shift_X = zeros(N,D,K);
    shift_Y = zeros(N,1,K);
    % ��Ȩ��С���˼���ع�ϵ��
    for k =1:K
    shift_X(:,:,k) = X.*repmat(pW(:,k),1,D);
    shift_Y(:,:,k) = Y.*repmat(pW(:,k),1,1);
    %���������ڵ������»ع�ϵ��betaʱ���������Է���ϵ������������ô�죡����
    B(:,k) = (shift_X(:,:,k)'*shift_X(:,:,k))\(shift_X(:,:,k)'*shift_Y(:,:,k));
    %���������ڵ������»ع�ϵ��betaʱ���������Է���ϵ������������ô�죡����
    end
    % check for convergence 
    if abs(Obj_Iter_Values(j+1) - Obj_Iter_Values(j)) < (threshold)
        disp(['��',num2str(j),'�ε������ܲв�ǰ��������ľ���ֵС��1e-6']);
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

%% ������ʼ��
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
    % ����ȷ�K�࣬��С���˷���ʼ��K���ع�ϵ��
    B(:,kk) = (x(:,:,kk)'*x(:,:,kk))\(x(:,:,kk)'*y(:,:,kk));
end
% where kk == K 
xx(:,:,K) = X(((K-1)*num+1):end,:);
yy(:,:,K) = Y(((K-1)*num+1):end,:);
B(:,K) = (xx(:,:,K)'*xx(:,:,K))\(xx(:,:,K)'*yy(:,:,K));

end


%% ����ģ������U_ik
function U_ik = fuzzy_u_ik(e_ik)
% �������õ� ��e_ik ����i��������k��ƽ������ƽ��
% U_ik ��ģ������U_ik ,N*K

U_ik = zeros(N,K);
m_1 = 1/(m - 1);
% ����ͳ��e_ki��ÿ��0Ԫ�صĸ��� e_ki_row_0
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
[U_row_max,max_id] = max(U_ik,[],2);
id_update = find(U_row_max>Alpha);
% ��֪����Alpha=1ʱ��U_ik������
nl =length(id_update);
%% �Գ���Alpha��Uik,����
if nl>0
    for s = 1:nl
%         U_ik(id_update(s),:) = (U_ik(id_update(s),:)==U_row_max(id_update(s)));
        U_ik(id_update(s),:) = 0;
        U_ik(id_update(s),max_id(id_update(s))) = 1;       
    end
end

end
%% ����e_ik���󣬵�i��������k�����ƽ��
function e_ik = error_ik(Beta)
% Beta: K����ƽ��ϵ����d*K
% e_ki:��i��������k�����ƽ��,N*K
% 
Y_K = repmat(Y, 1, K); 
XB = X*Beta;
gap = Y_K - XB;
e_ik = gap.*gap; % ��i��������k�����ƽ��

end

function total_residule = calcu_residule(B,U_ik)
% �����ܵĲв�ƽ����
resid = zeros(K,1);
for jj = 1:K
    Bk = B(:,jj);
    pW_k = diag(U_ik(:,jj)).^m;
    %%%%%%%%%%%%%%%%%%%%%%%%%%% �ɽ�pGamma_k�Ȳ���Ȩ���㣬�������
    resid(jj) = (Y-X*Bk)'*pW_k*(Y-X*Bk);     
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
end
total_residule = sum(resid);  % ������������������ƽ��Ĳв��
end

end