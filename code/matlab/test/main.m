clear all
clc
num=xlsread('F:\����ũҵ��ѧ����ʦ��\��Ҷ\��Ҷ����\ʵ����Ŀ\����\Ԥ��������\������\Ҷ����a1234D2_test.xlsx');
[r,c]=size(num)
Xcal= num(1:141,2:161);
ycal = num(1:141, 1);
xaxis=num(142,2:161);
Model=ipls(Xcal,ycal,8,'mean',10,xaxis,'syst123',5)
iplsplot(Model,'wavlabel')


