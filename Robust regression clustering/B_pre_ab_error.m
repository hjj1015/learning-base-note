function B_PRE_AB_ERROR = B_pre_ab_error(B_pre,B_True)
%   ����Ԥ������Ļع�ϵ��B_pre����ʵB_True�ľ������
B_PRE_AB_ERROR = norm(B_pre -B_True);
