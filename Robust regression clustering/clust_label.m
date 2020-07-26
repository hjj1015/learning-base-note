function [label,data1,data2] = clust_label(x,y,B1,B2)
% �����Ѿ�Ԥ����Ļع�ϵ���������ݷ���
% y\in R^n
% �����ܵ�������
n= length(y);

dist1 = abs(y-x*B1);
dist2 = abs(y-x*B2);
label = ones(n,1);
label(dist2<=dist1) = 2;
x(:,end) = [];
data1 = [x(label==1),y(label==1)];
data2 = [x(label==2),y(label==2)];

end