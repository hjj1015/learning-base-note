function varargout = FCRM_entropy(data,K,entropy_coef,B_initial)
% goal:ʹ��L1�����ȣ�������ͷ���ģ������ϻع�����
% {x_i,y_i} from K hyperplanes with parametres B1,B2��...,B_k 
% data:(x_i,y_i),x_i \in R^{1*dim},���һ��ȫΪ1
%                y_i \in R 
%                Sample_size = N
%   entropy_coef:�سͷ����ϵ��>0������ڣ���ʷ����entropy_coef=0.9898��0.7�����õ��Ƕ�������
%   B_initial�� K����ʼ�ĳ�ƽ��ϵ����d*K
% �����FCMģ�͵Ĳ����û��ģ������m������
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

%% ͣ����ʽѡ��Ŀ�꺯����j+1�ε���ֵ���j�ε���Ŀ��ֵ�Ĳ���,���������������10000
Obj_Iter_Values = zeros(1000,1);  % Ԥ�����ڴ棬�洢ÿ��������Ŀ�꺯��ֵ
t = tic;
j = 1 ;  % ����������������10000
while j<1000
    total_residule = calcu_residule(x,y,B,U_ik);
    Obj_Iter_Values(j+1) = total_residule;
    pW = U_ik.^m;

    % new value for parameters of each Component
    Nk = sum(U_ik, 1);   % pW ������ͣ���1*K�ľ���
    pPi = Nk/N;            %�����pPiҲ�ǣ�1��K����С,�Ǹ���������ռ���������ĸ���
    shift_X = zeros(N,D,K);
    shift_Y = zeros(N,1,K);
    % ��Ȩ��С���˼���ع�ϵ��
    for k =1:K
    shift_X(:,:,k) = x.*repmat(pW(:,k),1,D);   % shift_X(:,:,k)��N*D��С
    shift_Y(:,:,k) = y.*repmat(pW(:,k),1,1);   % shift_Y(:,:,k)��N*1��С
    B(:,k) = (shift_X(:,:,k)'*shift_X(:,:,k))\(shift_X(:,:,k)'*shift_Y(:,:,k));
%     %% ADMM�㷨���|y-x*Bk|
%     B(:,k) = lad(shift_X(:,:,k), shift_Y(:,:,k), 1.0, 1);
    end
    % check for convergence 
    if abs(Obj_Iter_Values(j+1) - Obj_Iter_Values(j)) < (threshold)
        disp(['��',num2str(j),'�ε������ܲв�ǰ��������ľ���ֵС��1e-12']);
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

%% ������ʼ��
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
    % ����ȷ�K�࣬��С���˷���ʼ��K���ع�ϵ��
    B(:,kk) = (xx(:,:,kk)'*xx(:,:,kk))\(xx(:,:,kk)'*yy(:,:,kk));
end
% where kk == K 
x2(:,:,K) = X(((K-1)*num+1):end,:);
y2(:,:,K) = Y(((K-1)*num+1):end,:);
B(:,K) = (x2(:,:,K)'*x2(:,:,K))\(x2(:,:,K)'*y2(:,:,K));

end


%% ����ģ������U_ik
function U_ik = fuzzy_u_ik(e_ik,penalty_coef)
% �������õ� ��e_ik ����i��������k��ƽ������ƽ��,N*K
% U_ik ��ģ������U_ik ,N*K
%
dis_lambda = -e_ik/penalty_coef;
% ����exp(-|dij|/penalty_coef),exp_dis : size,N*K
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
% sum_exp = sum(exp_dis,2); % ���ж�exp_dis��ͣ�N*1
% rep_exp = repmat(sum_exp,1,K);
% % �������յ�ģ������
% U_ik = exp_dis./rep_exp;

end
%% ����e_ik���󣬵�i��������k�����
function e_ik = error_ik(X,Y,Beta)
% Beta: K����ƽ��ϵ����d*K
% e_ki:��i��������k�����ƽ��,N*K
% 
Y_K = repmat(Y, 1, K); 
XB = X*Beta;
gap = Y_K - XB;
e_ik = abs(gap);  % ��i��������k�����,L1����
e_ik = e_ik.*e_ik;% ��i��������k�����,L2����
end

function total_residule = calcu_residule(x,y,B,U_ik)
% �����ܵĲв�ƽ����
resid = zeros(K,1);
for jj = 1:K
    Bk = B(:,jj);
    pW_k = diag(U_ik(:,jj)).^m;
    inter_dis = pW_k*abs(y-x*Bk);  % N*1����������ֵ������Ȩ��
    %%%%%%%%%%%%%%%%%%%%%%%%%%% �ɽ�pGamma_k�Ȳ���Ȩ���㣬�������
    resid(jj) = sum(inter_dis);     
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
end
total_residule = sum(resid);  % ������������������ƽ��Ĳв��
end

end