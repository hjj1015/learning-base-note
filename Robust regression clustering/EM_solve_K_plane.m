function varargout = EM_solve_K_plane(X,Y,K,B_initial)
%   X :(n1+n2)*dim
%   Y:(n1+n2)*1
%   {x_i,y_i} from K hyperplanes with parametres B1,B2��...,B_k 
%   i = 1,2,...(n1+n2)
%   K����ƽ��
%   B_initial�� K����ʼ�ĳ�ƽ��ϵ����d*K

% ͣ�������
threshold = 1e-6; 

% ��ȡ������N������ά��D
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

%% ͣ����ʽѡ��Ŀ�꺯����j+1�ε���ֵ���j�ε���Ŀ��ֵ�Ĳ���,���������������10000
Obj_Iter_Values = zeros(1000,1);  % Ԥ�����ڴ棬�洢ÿ��������Ŀ�꺯��ֵ

t = tic;
j = 1 ;  % ����������������10000

while j<1000
    [Px,total_residule] = calc_prob();
    Obj_Iter_Values(j+1) = total_residule;
    
    pGamma = Px .* repmat(pPi, N, 1);  % pGamma��N*K   
    % pGamma��N*K�����Ⱦ��󣬵�N���������ڵ�K���ص�������
    pGamma = pGamma ./ repmat(sum(pGamma, 2), 1, K);  

    % new value for parameters of each Component
    Nk = sum(pGamma, 1);   % pGamma ������ͣ���1*K�ľ���
    pPi = Nk/N;            %�����pPiҲ�ǣ�1��K����С,�Ǹ���������ռ���������ĸ���
    shift_X = zeros(N,D,K);
    shift_Y = zeros(N,1,K);
    Sigma =  zeros(K,1); %k���sigma����
    for k =1:K
    shift_X(:,:,k) = X.*repmat(pGamma(:,k),1,D);
    shift_Y(:,:,k) = Y.*repmat(pGamma(:,k),1,1);
    %��������������������������������������������������������������������������
    B(:,k) = (shift_X(:,:,k)'*shift_X(:,:,k))\(shift_X(:,:,k)'*shift_Y(:,:,k));
    %��������������������������������������������������������������������������
    % ���µ�k���sigma2����
    Xt = Y - X*B(:,k);
    Sigma(k,1) = (Xt' * ...
        (diag(pGamma(:, k)) * Xt)) / Nk(k);
    end
    % check for convergence 
    if abs(Obj_Iter_Values(j+1) - Obj_Iter_Values(j)) < (threshold)
        disp(['��',num2str(j),'�ε������ܲв�ǰ��������ľ���ֵС��1e-12']);
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

%% ������ʼ��
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
    % ����ȷ�K�࣬��С���˷���ʼ��K���ع�ϵ��
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

%% EM������ⳬƽ��ĸ���ʱ
function [Px,total_residule] = calc_prob()
Px = zeros(N, K);
% total_residule: N*1, ��i�д洢���ǵ�i������������ƽ��Ĳв�ľ���ֵ�ĺ�
residul = zeros(K,1);
for k = 1:K
    Bk = B(:,k);
    pGamma_k = diag(pGamma(:,k));
    residul(k) = (Y-X*Bk)'*pGamma_k*(Y-X*Bk);     
end
% Px:N*K, λ��(i,j)ָ��i�������ڵ�j����ƽ��Ĳв����
for kk = 1:K
    Xt = Y - X*B(:,kk);
    tmp = (Xt.*Xt)/Sigma(kk);
    coef = (2*pi)^(-1/2)/sqrt(Sigma(kk));
    Px(:, kk) = coef * exp(-0.5*tmp);
end
total_residule = sum(residul);  % ������������������ƽ��Ĳв��
end


end


