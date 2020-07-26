function [B_step_err,j,model] = EM_B_error_trace(x,y,B_TRUE,B_initial)

threshold = 1e-15;
[N, D] = size(x);
B1_intial = B_initial(:,1);
B2_intial = B_initial(:,2);
B = [B1_intial,B2_intial];
B1_True = B_TRUE(:,1);
B2_True = B_TRUE(:,2);

%% initial values

[pMiu,pPi,psigma] = init_params();
% Update  
B_1_before = B1_intial;
t = tic;
j = 1 ;
total_residule_before = 0;
B_ab_err = zeros(1000,1);  % 预分配内存，存B每步的绝对误差
B_re_err = zeros(1000,1);  % 预分配内存，存B每步的相对误差

while true
	[Px,~] = calc_prob();    
    % new value for parameters of each Component
    Nk = sum(Px, 1);   % pGamma 按列求和，得1*K的矩阵
    pPi = Nk/N;    %这里的pPi也是（1，K）大小,是各簇中样本占总样本量的概率
    shift_X = zeros(N,D,2);
    shift_Y = zeros(N,1,2);
    % update B
    for k =1:2
        shift_X(:,:,k) = x.*repmat(Px(:,k),1,D);  %/Nk(k)，这个改变了不知道对不对，也不知道效果是变好还是，要试试
        shift_Y(:,:,k) = y.*repmat(Px(:,k),1,1);
        B(:,k) = (shift_X(:,:,k)'*shift_X(:,:,k))\(shift_X(:,:,k)'*shift_Y(:,:,k));
    end
    [B1_regress,B2_regress] = identify_B(B(:,1),B(:,2),B1_True,B2_True);
    B_ab_err(j) = (B_pre_ab_error(B1_regress,B1_True)+B_pre_ab_error(B2_regress,B2_True))/2;
    B_re_err(j) = (B_relative_error(B1_regress,B1_True)+B_relative_error(B2_regress,B2_True))/2;
    pMiu = x*B;
    % 更新psigma
    psigma = sum(Px.*(repmat(y, 1, 2)-x*B).*(repmat(y, 1, 2)-x*B))./Nk; 
%     distmat = repmat(y.*y, 1, 2) + ...
%         pMiu.*pMiu  - 2*[y,y].*pMiu;
%     [~ ,labels] = min(distmat, [], 2);
%     for k=1:2
%         yk = y(labels == k, :)-x(labels == k, :)*B(:,k);
%         psigma(k) = cov(yk);
%     end
    psigma = max(psigma,0.03);%控住sigma值，不能小于0.01
    % check for convergence
%     B_1 = B(:,1);
%     if norm(B_1 -B_1_before) < threshold
%         disp(['第',num2str(j),'次迭代前后两步系数B1的差距二范数小于1e-15']);
%         disp(['平均每个维度上的前后差距为:',num2str(norm(B_1 -B_1_before)/D)]);
%         break;
%     end
%     if sum(abs(total_residule - total_residule_before)) < (threshold*1000)
%         disp(['第',num2str(j),'次迭代的总残差前后两步差的绝对值小于1e-12']);
%         break;
%     end
%     total_residule_before = total_residule ;
%     B_1_before = B_1;
    %   if code run more than 10s ,stop it whether or not convergent
    if j > 50
        break;
        disp('程序时间超50s，循环终止');
    end
    j = j + 1;
end

B_step_err = [B_ab_err,B_re_err];  % j*2
model.B = B;

function [pMiu,pPi,psigma] = init_params()
pMiu = x*[B1_intial,B2_intial]; %N*2   miu_ik,i=1...N; k=1,2

% pPi = zeros(1, 2);
pPi = [0.5,0.5];
psigma = zeros(1,2);
% hard assign x to each centroids
distmat = repmat(y.*y, 1, 2) + ...
	pMiu.*pMiu  - 2*[y,y].*pMiu;
[~ ,labels] = min(distmat, [], 2);
for k=1:2
    yk = y(labels == k, :);
    pPi(k) = size(yk, 1)/N;
    psigma(k) = cov(yk); 
end
end

function [Px,total_residule] = calc_prob()
Px = zeros(N, 2);
total_residule = sum(abs(repmat(y,1,2)-x*B),2); % total_residule:N*1,第i行存储的是第i个样本到所有平面的残差的绝对值的和
for k = 1:2
    yshift = y-pMiu(:,k);
    inv_pSigma = 1/(psigma(k));
    tmp = (yshift*inv_pSigma) .* yshift;
    coef = (2*pi)^(-1/2) * sqrt(inv_pSigma);
    Px(:, k) = coef * exp(-0.5*tmp);
end
px = Px.*repmat(pPi,N,1);
Px = px./repmat(sum(px,2),1,2);
total_residule = sum(total_residule);  % 计算所有样本到所有平面的残差和
end

end