function Frate = fluctuation(Cluster_Center)
% Cluster_Center:簇中心，size：K*p
% Frate:每个簇的波动率，K*1

[K,~] = size(Cluster_Center);
Frate = zeros(K,1);
for i = 1:K
    Ci = Cluster_Center(K,:);
    Frate(i,1) = var(Ci);
end

end