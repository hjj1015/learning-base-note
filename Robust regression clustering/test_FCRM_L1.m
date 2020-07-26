function varargout = test_FCRM_L1(data,K,Beta_ini_list,iter_max,beta1,beta2,dim)
% ����һ����ʼ���ع�ϵ��Beta_ini_list��kmeans�ع�����Ч��
% beta1,beta2��ʵ�Ļع�ϵ��
% ���һ���ع�����Ļع�ϵ��B_kmeans���Լ�������
if nargin<7
    % beta��ά����2
    dim = 2;
end
B_kmeans = zeros(2*dim,iter_max);
B_relative_err = zeros(iter_max,1);
B_true = [beta1;beta2];
B_nm = norm(B_true);
m = 2;% ģ������
pi1 = zeros(iter_max,1);
for test_n = 1:iter_max
    Beta_inital = reshape(Beta_ini_list(:,test_n),dim,2);
    [~,FCRM_model] = L1_FCRM(data,K,m,Beta_inital);
    B_hat_FCRM = FCRM_model.B;
    pPi = FCRM_model.pPi;
    pi1(test_n) = min(pPi);

    [B1_FCRM,B2_FCRM] = identify_B(B_hat_FCRM(:,1),B_hat_FCRM(:,2),beta1,beta2);
    B_kmeans(:,test_n) = [B1_FCRM;B2_FCRM];
    B_gap = B_kmeans(:,test_n) - B_true;
    B_relative_err(test_n) = norm(B_gap)/B_nm;
end
B_kmeans = B_kmeans';
if nargout==3
    % beta��ά����2
   varargout= {B_kmeans,B_relative_err,pi1};
else
    varargout= {B_kmeans,B_relative_err};
end

end
    



