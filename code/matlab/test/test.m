clc
clear all
load nirbeer; 
Model=ipls(Xcal,ycal,10,'mean',20,xaxis,'syst123',5);
iplsplot(Model,'intlabel')