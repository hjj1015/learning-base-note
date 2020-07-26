function B_PRE_AB_ERROR = B_pre_ab_error(B_pre,B_True)
%   计算预测出来的回归系数B_pre与真实B_True的绝对误差
B_PRE_AB_ERROR = norm(B_pre -B_True);
