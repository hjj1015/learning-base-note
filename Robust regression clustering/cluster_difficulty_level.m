function CDL = cluster_difficulty_level(x,y,B_True,lambda)
%     CDL:�ṹ�壬���ڿ̻���ͬ�����ľ������ѳ̶ȣ���������Dÿ�еľ�ֵ���ͷ��Ȼ�����������
%     mu ��Dÿ�������ľ�ֵ
%     sigma:Dÿ�������ķ���,ҪԽ��Ҫ�ã�����ӳ��ÿ����������ͬ�ĴصĲ�����
%     D: ԭʼn�������������ع�ϵ���ľ������D������ֵ���룩
% lambda: ��ӳ�������������صľ����ֵ�ͷ��Ӱ����������Ѷȵĳ̶�
% B_True �� ����ֻ����K=2�����Σ�B1,B2
if nargin <4
    %mu,sigmaӰ���������̶���ͬ��һ��ʵ��mu��sigma�ƺ�Ӱ����󣬹�һ��lambda<1
    lambda = 1;
end

[~,k] = size(B_True);
D = repmat(y,1,k) - x*B_True;  % N*K
D = abs(D);

mu = mean(D,2);  % ���м����ֵ, N*1
sigma = var(D,0,2);% ���м��㷽��
% lambda>1ʱ����ʾsigma����Ҫ��һ��lambda<1��mu����Ҫ��Խ�����ؾ���С������������Խ�ã�
% mu��sigma�ĳ˻���ʾcdl,N*1��N�������������÷֣�muС��sigma��cdl�÷�Խ����������Խ��
cdl = lambda*sigma./mu;  
% cdl2 = abs(mu - lambda*sigma);
score = sum(cdl);

CDL.D = D;
CDL.mu = mu;
CDL.sigma = sigma;
CDL.cdl = cdl;
CDL.score = score;






