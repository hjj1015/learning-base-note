clc;clear all;
%% plot entropy term 
u = 0.001:0.001:1;
y = -u.*log(u);
plot(u,y,'b-');
grid on
hold on
plot(exp(-1),exp(-1),'ro');