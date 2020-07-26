function [label,data1,data2] = clust_label(x,y,B1,B2)
% 根据已经预测出的回归系数，对数据分类
% y\in R^n
% 两类总的样本量
n= length(y);

dist1 = abs(y-x*B1);
dist2 = abs(y-x*B2);
label = ones(n,1);
label(dist2<=dist1) = 2;
x(:,end) = [];
data1 = [x(label==1),y(label==1)];
data2 = [x(label==2),y(label==2)];

end