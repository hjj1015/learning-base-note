function [B1_regress,B2_regress] = identify_B(B1_pre,B2_pre,B1_True,B2_True)
%   识别预测出的系数B1，B2
b_true = [B1_True;B2_True];
b_pre1 = [B1_pre;B2_pre];
b_pre2 = [B2_pre;B1_pre];

B1_gap = b_pre1-b_true;
B2_gap = b_pre2-b_true;

B1_rela = norm(B1_gap);
B2_rela = norm(B2_gap);
if B1_rela < B2_rela
    B1_regress = B1_pre;
    B2_regress = B2_pre;
else
    B1_regress = B2_pre;
    B2_regress = B1_pre;
end
