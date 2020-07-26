function CDL = cluster_difficulty_level(x,y,B_True,lambda)
%     CDL:结构体，用于刻画不同样本的聚类困难程度，求距离矩阵D每行的均值，和方差；然后再整体计算
%     mu ：D每行向量的均值
%     sigma:D每行向量的方差,要越大要好，它反映了每个样本到不同的簇的差异性
%     D: 原始n个样本到各个回归系数的距离矩阵D（绝对值距离）
% lambda: 反映各样本到各个簇的距离均值和方差，影响聚类任务难度的程度
% B_True ： 这里只考察K=2类情形，B1,B2
if nargin <4
    %mu,sigma影响聚类任务程度相同，一般实际mu比sigma似乎影响更大，故一般lambda<1
    lambda = 1;
end

[~,k] = size(B_True);
D = repmat(y,1,k) - x*B_True;  % N*K
D = abs(D);

mu = mean(D,2);  % 按行计算均值, N*1
sigma = var(D,0,2);% 按行计算方差
% lambda>1时，表示sigma更重要；一般lambda<1（mu更重要，越靠近簇距离小的样本，质量越好）
% mu与sigma的乘积表示cdl,N*1，N个样本的质量得分；mu小，sigma大，cdl得分越大，样本质量越好
cdl = lambda*sigma./mu;  
% cdl2 = abs(mu - lambda*sigma);
score = sum(cdl);

CDL.D = D;
CDL.mu = mu;
CDL.sigma = sigma;
CDL.cdl = cdl;
CDL.score = score;






