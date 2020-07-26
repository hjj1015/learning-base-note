clc;clear all;
%% 读取数据Iris: [sepal_length, sepal_width, petal_length, petal_width]
[A,B,Class] = textread('banana.txt','%f%f%f','delimiter',',');
data=[A,B];
n = size(data,1);
%% 映射簇标签:CL
CL = zeros(n,1);
CL(Class==-1) = 0;
CL(Class==1) = 1;
%% plot(Irisdata):三维vs二维
figure(1)
plot(A(Class==-1),B(Class==-1),'ro'),grid on
hold on
plot(A(Class==1),B(Class==1),'go')
hold off
% 
figure(2)
plot3(A(1:50),B(1:50),C(1:50),'ro'),grid on
hold on
plot3(A(51:100),B(51:100),C(51:100),'go')
plot3(A(101:150),B(101:150),C(101:150),'bo')
hold off
% legend('Iris-setosa','Iris-versicolor','Iris-virginica');
title('IRIS True Label');
%% AP clustering for Iris
% 构造相似矩阵
s  = zeros(n);
o  = nchoosek(1:n,2);      % set up all possible pairwise comparisons 
xx  = data(o(:,1),:)';           % point 1
xy  = data(o(:,2),:)';           % point 2
d2  = (xx - xy).^2;           % distance squared
d  = -sqrt(sum(d2));          % distance

k  = sub2ind([n n], o(:,1), o(:,2) );    % prepare to make square 
s(k) = d;             
s = s + s';
di = 1:(n+1):n*n;         %index to diagonal elements
s(di) = 5*min(d);   % Preference,设置为样本间距离相似度s最小的值，做为距离相似度S的对角元素
[ex,AA,R] = affprop(s);
cluter_idx = unique(ex);
figure(3)
color = ['r','g','b','c','m','y','w'];
% plot3(data(:,1),data(:,2),data(:,3),'o'),
for k =1:length(cluter_idx)
    xk = data(ex==cluter_idx(k),:);
    plot3(xk(:,1),xk(:,2),xk(:,3),[color(k),'o'])
    hold on
end
hold off,grid on,title('Affinity Propagation Clustering');
%% kmeans
n_clusters = 3;
tic;
[idx,Cc] = kmeans(data,n_clusters);
t_kmeans = toc(tic);
cluter_idx = unique(idx);
figure(4)
color = ['g','b','r','c','m','y','w'];
for k =1:length(cluter_idx)
    xk = data(idx==cluter_idx(k),:);
    plot3(xk(:,1),xk(:,2),xk(:,3),[color(k),'o'])
    hold on
end
hold off,grid on,title('Kmeans Clustering');
%% Fuzzy c-means clustering
tic;
[center,U] = fcm(data,n_clusters);
t_fcm = toc(tic);
maxU = max(U); 
figure(5)
color = ['r','g','b','c','m','y','w'];
for k =1:length(cluter_idx)
    index_k = find(U(k,:) == maxU); 
    xk = data(index_k,:);
    plot3(xk(:,1),xk(:,2),xk(:,3),[color(k),'o'])
    hold on
end
hold off,grid on,title('Fuzzy c-means Clustering');