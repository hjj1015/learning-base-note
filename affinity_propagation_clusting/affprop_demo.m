%% demonstrate affinity propagation
% 
% demonstration shows a movie while the message passaging is going on. 
% The first panel shows 2 dimensional data with a blue line connecting an
% exemplar (red point) to its constituents.  The remaining three panels
% show the messages being past. The top right shows the message sent from
% each potential exemplar (column) to the other points (rows). Brigher
% colors indicate higher availability. The lower left shows the
% responsibility matrix. This is a message sent from a point (row) to each 
% potential exemplar (column) indicating the suitability of that exemplar.
% The lower right is the combination of the two matrices and it indicates
% the current state. The column that maximizes a row is the examplar for
% that row. 
%
% Example
%    affprop_demo
% 
% See also affprop 

%% setup clustering
% much easier with stats toolbox because pdist is available
clc;clear all;
% generate 2d data
M = [5 0; 0 5; 10 10];
idx = repmat( 1:3, 50, 1 );
x = M(idx(:),:) + randn(150,2);
%% test_dataset
plot(x(:,1),x(:,2),'o'),title('raw\_data'),grid on
%% 依据负平方距离,构造相似矩阵s
% generate similarity matrix
m  = size(x,1);
s  = zeros(m);
o  = nchoosek(1:150,2);      % set up all possible pairwise comparisons 
xx  = x(o(:,1),:)';           % point 1
xy  = x(o(:,2),:)';           % point 2
d2  = (xx - xy).^2;           % distance squared
d  = -sqrt(sum(d2));          % distance

k  = sub2ind([m m], o(:,1), o(:,2) );    % prepare to make square 
s(k) = d;             
s = s + s';
di = 1:(m+1):m*m;         %index to diagonal elements

s(di) = min(d);   % Preference,设置为样本间距离相似度s最小的值，做为距离相似度S的对角元素 median(d)
%% AP Clustering
% options.StallIter = 10;
% options.OutputFcn = @(a,r) affprop_plot(a,r,x,'k.');
% 
% figure
% ex = affprop(s,options);
tic;
[ex,A,R] = affprop(s);
t_AP = toc(tic);
cluter_idx = unique(ex);
figure(2)
color = ['r','g','b','c','m','y','w'];
plot(x(:,1),x(:,2),'ko'),hold on
for k =1:length(cluter_idx)
    xk = x(ex==cluter_idx(k),:);
    plot(xk(:,1),xk(:,2),[color(k),'o'])
end
hold off,grid on,title('Affinity Propagation Clustering');
%% kmeans
tic;
[idx,C] = kmeans(x,3);
t_kmeans = toc(tic);
cluter_idx = unique(idx);
figure(3)
color = ['r','g','b','c','m','y','w'];
plot(x(:,1),x(:,2),'ko'),hold on
for k =1:length(cluter_idx)
    xk = x(idx==cluter_idx(k),:);
    plot(xk(:,1),xk(:,2),[color(k),'o'])
end
hold off,grid on,title('Kmeans Clustering');
%% Fuzzy c-means clustering
tic;
[center,U] = fcm(x,3);
t_fcm = toc(tic);
maxU = max(U); 
figure(4)
color = ['r','g','b','c','m','y','w'];
plot(x(:,1),x(:,2),'ko'),hold on
for k =1:length(cluter_idx)
    index_k = find(U(k,:) == maxU); 
    xk = x(index_k,:);
    plot(xk(:,1),xk(:,2),[color(k),'o'])
end
hold off,grid on,title('Fuzzy c-means Clustering');
fprintf('Affinity Propagation Clustering time used:%3.8f\n',t_AP);
fprintf('Kmeans Clustering time used:%3.8f\n',t_kmeans);
fprintf('Fuzzy c-means Clustering time used:%3.8f\n',t_fcm);



