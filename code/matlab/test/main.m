clear all
clc
num=xlsread('F:\华南农业大学龙老师组\茶叶\茶叶茶多酚\实验项目\数据\预处理数据\区波段\叶绿素a1234D2_test.xlsx');
[r,c]=size(num)
Xcal= num(1:141,2:161);
ycal = num(1:141, 1);
xaxis=num(142,2:161);
Model=ipls(Xcal,ycal,8,'mean',10,xaxis,'syst123',5)
iplsplot(Model,'wavlabel')


