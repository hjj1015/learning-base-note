function y = angle_calcu(a,b)
% �������������ļн�
% a,b������������
y = acos(a'*b/norm(a)/norm(b));
y = y*180/pi;
if y>90
    y = 180 - y;
end   

end
