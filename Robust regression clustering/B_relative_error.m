function B_relat_error = B_relative_error(B_initial,B_true)
%   B_initial:��ʼ����B
%   B_true:��ʵ��B
%   B_relative_error��������ʼ����������
B_relat_error = norm(B_initial -B_true)/norm(B_true);



