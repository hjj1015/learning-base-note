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
B_ab_err = zeros(1000,1);  % Ԥ�����ڴ棬��Bÿ���ľ������
B_re_err = zeros(1000,1);  % Ԥ�����ڴ棬��Bÿ����������

while true
	[Px,~] = calc_prob();    
    % new value for parameters of each Component
    Nk = sum(Px, 1);   % pGamma ������ͣ���1*K�ľ���
    pPi = Nk/N;    %�����pPiҲ�ǣ�1��K����С,�Ǹ���������ռ���������ĸ���
    shift_X = zeros(N,D,2);
    shift_Y = zeros(N,1,2);
    % update B
    for k =1:2
        shift_X(:,:,k) = x.*repmat(Px(:,k),1,D);  %/Nk(k)������ı��˲�֪���Բ��ԣ�Ҳ��֪��Ч���Ǳ�û��ǣ�Ҫ����
        shift_Y(:,:,k) = y.*repmat(Px(:,k),1,1);
        B(:,k) = (shift_X(:,:,k)'*shift_X(:,:,k))\(shift_X(:,:,k)'*shift_Y(:,:,k));
    end
    [B1_regress,B2_regress] = identify_B(B(:,1),B(:,2),B1_True,B2_True);
    B_ab_err(j) = (B_pre_ab_error(B1_regress,B1_True)+B_pre_ab_error(B2_regress,B2_True))/2;
    B_re_err(j) = (B_relative_error(B1_regress,B1_True)+B_relative_error(B2_regress,B2_True))/2;
    pMiu = x*B;
    % ����psigma
    psigma = sum(Px.*(repmat(y, 1, 2)-x*B).*(repmat(y, 1, 2)-x*B))./Nk; 
%     distmat = repmat(y.*y, 1, 2) + ...
%         pMiu.*pMiu  - 2*[y,y].*pMiu;
%     [~ ,labels] = min(distmat, [], 2);
%     for k=1:2
%         yk = y(labels == k, :)-x(labels == k, :)*B(:,k);
%         psigma(k) = cov(yk);
%     end
    psigma = max(psigma,0.03);%��סsigmaֵ������С��0.01
    % check for convergence
%     B_1 = B(:,1);
%     if norm(B_1 -B_1_before) < threshold
%         disp(['��',num2str(j),'�ε���ǰ������ϵ��B1�Ĳ�������С��1e-15']);
%         disp(['ƽ��ÿ��ά���ϵ�ǰ����Ϊ:',num2str(norm(B_1 -B_1_before)/D)]);
%         break;
%     end
%     if sum(abs(total_residule - total_residule_before)) < (threshold*1000)
%         disp(['��',num2str(j),'�ε������ܲв�ǰ��������ľ���ֵС��1e-12']);
%         break;
%     end
%     total_residule_before = total_residule ;
%     B_1_before = B_1;
    %   if code run more than 10s ,stop it whether or not convergent
    if j > 50
        break;
        disp('����ʱ�䳬50s��ѭ����ֹ');
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
total_residule = sum(abs(repmat(y,1,2)-x*B),2); % total_residule:N*1,��i�д洢���ǵ�i������������ƽ��Ĳв�ľ���ֵ�ĺ�
for k = 1:2
    yshift = y-pMiu(:,k);
    inv_pSigma = 1/(psigma(k));
    tmp = (yshift*inv_pSigma) .* yshift;
    coef = (2*pi)^(-1/2) * sqrt(inv_pSigma);
    Px(:, k) = coef * exp(-0.5*tmp);
end
px = Px.*repmat(pPi,N,1);
Px = px./repmat(sum(px,2),1,2);
total_residule = sum(total_residule);  % ������������������ƽ��Ĳв��
end

end