clear all
clc
[num]=xlsread('F:\����ũҵ��ѧ����ʦ��\��Ҷ\��Ҷ����\ʵ����Ŀ\����\dtest\�ɼ�+������\ipls\Ҷ����a23_test.xlsx');
Xcal= num(1:141,2:75);
ycal = num(:, 1)
xaxis=num(142,2:75)
Model=ipls(Xcal,ycal,5,'mean',10,xaxis,'syst123',5)
iplsplot(Model,'intlabel')