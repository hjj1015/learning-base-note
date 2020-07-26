function Frate = fluctuation(Cluster_Center)
% Cluster_Center:�����ģ�size��K*p
% Frate:ÿ���صĲ����ʣ�K*1

[K,~] = size(Cluster_Center);
Frate = zeros(K,1);
for i = 1:K
    Ci = Cluster_Center(K,:);
    Frate(i,1) = var(Ci);
end

end