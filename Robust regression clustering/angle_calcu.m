function y = angle_calcu(a,b)
% 计算两个向量的夹角
% a,b是两个列向量
y = acos(a'*b/norm(a)/norm(b));
y = y*180/pi;
if y>90
    y = 180 - y;
end   

end
