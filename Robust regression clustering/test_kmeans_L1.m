function varargout = test_kmeans_L1(data,K,Beta_ini_list,iter_max,beta1,beta2,dim)
% 检验一串初始化回归系数Beta_ini_list，kmeans回归聚类的效果
% beta1,beta2真实的回归系数
% 输出一串回归聚类后的回归系数B_kmeans，以及相对误差
if nargin<7
    % beta的维数是2
    dim = 2;
end
B_kmeans = zeros(2*dim,iter_max);
B_relative_err = zeros(iter_max,1);
B_true = [beta1;beta2];
B_nm = norm(B_true);
pi1 = zeros(iter_max,1);
for test_n = 1:iter_max
    Beta_inital = reshape(Beta_ini_list(:,test_n),dim,2);
    try
    [~,FCRM_model] = L1_Hard_SOLVE(data,K,Beta_inital);
    B_hat_FCRM = FCRM_model.B;
    pPi = FCRM_model.pPi;
    pi1(test_n) = min(pPi);

    [B1_FCRM,B2_FCRM] = identify_B(B_hat_FCRM(:,1),B_hat_FCRM(:,2),beta1,beta2);
    B_kmeans(:,test_n) = [B1_FCRM;B2_FCRM];
    B_gap = B_kmeans(:,test_n) - B_true;
    B_relative_err(test_n) = norm(B_gap)/B_nm;
    catch
        pi1(test_n) = 0;
        B_kmeans(:,test_n) = zeros(1,2*dim);
        B_relative_err(test_n) = 100;
    end
end
B_kmeans = B_kmeans';
if nargout==3
    % beta的维数是2
   varargout= {B_kmeans,B_relative_err,pi1};
else
    varargout= {B_kmeans,B_relative_err};
end
end
    



