function B_relat_error = B_relative_error(B_initial,B_true)
%   B_initial:初始化的B
%   B_true:真实的B
%   B_relative_error：迭代初始化的相对误差
B_relat_error = norm(B_initial -B_true)/norm(B_true);



