clear all;
clc

hong = xlsread('D:\tea\含量\含量\allgai.xlsx');
hong2 = importdata('D:\tea\含量\含量\新建文本文档.txt');


h2 = hong(1:9,1:5);
h3 = hong(10:35,1:5);
h4 = hong(36:45,1:5);

b2 = hong(46:55,1:5);
b3 = hong(56:70,1:5);
b4 = hong(71:80,1:5);

f2 = hong(81:105,1:5);
f3 = hong(106:125,1:5);
f4 = hong(126:140,1:5);

we = ones(1,140);
%定义标签
% 
%绘图
h = figure;
set(h,'position',[250 150 650 800]);
subplot(5,1,1)
h1 = heatmap(hong(:,1)');
colormap(h1, 'pink');
subplot(5,1,2)
h2 = heatmap(hong(:,2)');
colormap(h2, 'pink');

subplot(5,1,3)
h3 = heatmap(hong(:,3)');
colormap(h3, 'pink');
subplot(5,1,4)
h4 = heatmap(hong(:,4)');
colormap(h4, 'pink');
subplot(5,1,5)
h5 = heatmap(hong(:,5)');
colormap(h5, 'pink');

% axis([-4,61,-4,20]);
% set(gca,'xtick',[],'ytick',[],'xcolor','w','ycolor','w');
% 'Chl a','Chl b','Carotenoids','GPT','Amino Acid'





