clear all;
clc

hong = xlsread('D:\tea\含量\含量\allgai.xlsx');
hong2 = importdata('D:\tea\含量\含量\新建文本文档.txt');


h2 = hong(1:9,1:5);h3 = hong(10:35,1:5);h4 = hong(36:45,1:5);
b2 = hong(46:55,1:5);b3 = hong(56:70,1:5);b4 = hong(71:80,1:5);
f2 = hong(81:105,1:5);f3 = hong(106:125,1:5);f4 = hong(126:140,1:5);

mean_h2 = mean(h2)*50000/31202;mean_h3 = mean(h3)*50000/28855;mean_h4 = mean(h4)*50000/33813;
mean_b2 = mean(b2)*50000/34044;mean_b3 = mean(b3)*50000/34650;mean_b4 = mean(b4)*50000/34642;
mean_f2 = mean(f2)*50000/34721;mean_f3 = mean(f3)*50000/32721;mean_f4 = mean(f4)*50000/33511;

max_h2 = max(h2);max_h3 = max(h3);max_h4 = max(h4);
max_b2 = max(b2);max_b3 = max(b3);max_b4 = max(b4);
max_f2 = max(f2);max_f3 = max(f3);max_f4 = max(f4);

min_h2 = min(h2);min_h3 = min(h3);min_h4 = min(h4);
min_b2 = min(b2);min_b3 = min(b3);min_b4 = min(b4);
min_f2 = min(f2);min_f3 = min(f3);min_f4 = min(f4);

meanwe = [mean_h2;mean_h3;mean_h4;mean_b2;mean_b3;mean_b4;mean_f2;mean_f3;mean_f4];
maxwe = [max_h2;max_h3;max_h4;max_b2;max_b3;max_b4;max_f2;max_f3;max_f4];
minwe = [min_h2;min_h3;min_h4;min_b2;min_b3;min_b4;min_f2;min_f3;min_f4];
a = 5;
h = figure;
set(h,'position',[250 150 950 350]);

bw = bar(1:9,[mean_h2(1:3)*a,mean_h2(4:5);mean_h3(1:3)*a,mean_h3(4:5);mean_h4(1:3)*a,mean_h4(4:5);
    mean_b2(1:3)*a,mean_b2(4:5);mean_b3(1:3)*a,mean_b3(4:5);mean_b4(1:3)*a,mean_b4(4:5);
    mean_f2(1:3)*a,mean_f2(4:5);mean_f3(1:3)*a,mean_f3(4:5);mean_f4(1:3)*a,mean_f4(4:5)]);hold on;

for i = 1:9
    plot([0.66-1+i,0.74-1+i],[maxwe(i,1)*5,maxwe(i,1)*5],'k','LineWidth',1);hold on;%绘制纵横线
    plot([0.66-1+i,0.74-1+i],[minwe(i,1)*5,minwe(i,1)*5],'k','LineWidth',1);hold on;%绘制纵横线
    plot([0.7-1+i,0.7-1+i],[minwe(i,1)*5,maxwe(i,1)*5],'k','LineWidth',1);hold on;%绘制纵横线

    plot([0.81-1+i,0.89-1+i],[maxwe(i,2)*5,maxwe(i,2)*5],'k','LineWidth',1);hold on;%绘制纵横线
    plot([0.81-1+i,0.89-1+i],[minwe(i,2)*5,minwe(i,2)*5],'k','LineWidth',1);hold on;%绘制纵横线
    plot([0.85-1+i,0.85-1+i],[minwe(i,2)*5,maxwe(i,2)*5],'k','LineWidth',1);hold on;%绘制纵横线

    plot([0.96-1+i,1.04-1+i],[maxwe(i,3)*5,maxwe(i,3)*5],'k','LineWidth',1);hold on;%绘制纵横线
    plot([0.96-1+i,1.04-1+i],[minwe(i,3)*5,minwe(i,3)*5],'k','LineWidth',1);hold on;%绘制纵横线
    plot([1-1+i,1-1+i],[minwe(i,3)*5,maxwe(i,3)*5],'k','LineWidth',1);hold on;%绘制纵横线

    plot([1.11-1+i,1.19-1+i],[maxwe(i,4),maxwe(i,4)],'k','LineWidth',1);hold on;%绘制纵横线
    plot([1.11-1+i,1.19-1+i],[minwe(i,4),minwe(i,4)],'k','LineWidth',1);hold on;%绘制纵横线
    plot([1.15-1+i,1.15-1+i],[minwe(i,4),maxwe(i,4)],'k','LineWidth',1);hold on;%绘制纵横线

    plot([1.26-1+i,1.34-1+i],[maxwe(i,5),maxwe(i,5)],'k','LineWidth',1);hold on;%绘制纵横线
    plot([1.26-1+i,1.34-1+i],[minwe(i,5),minwe(i,5)],'k','LineWidth',1);hold on;%绘制纵横线
    plot([1.3-1+i,1.3-1+i],[minwe(i,5),maxwe(i,5)],'k','LineWidth',1);hold on;%绘制纵横线

end
plot([0,10],[29.95,29.95],'k','LineWidth',1);
% plot([0.5,9.5],[5,5]);
yyaxis left
%修改横坐标名称、字体
set(gca,'XTickLabel',{'H2','H3','H4','B2','B3','B4','F2','F3','F4'},'FontSize',12,'FontName','Arial');
set(gca,'YTickLabel',{'0','1','2','3','4','5','6'},'FontSize',12,'FontName','Arial');
set(gca,'FontSize',12,'FontName','Arial');
xlabel({'\fontname{Arial}Tea Varieties and Leaf Positions'},'FontSize',12,'LineWidth',1);
ylabel({'\fontname{Arial}Content (%)'},'FontSize',12,'LineWidth',1);
set(gca,'FontSize',12,'Fontname', 'Arial','LineWidth',1);
set(gca,'tickdir','out');
box off
% 'Chl a','Chl b','Carotenoids','GPT','Amino Acid'

yyaxis right
set(gca,'YTickLabel',{'0','5','10','15','20','25'},'FontSize',12,'FontName','Arial');
set(gca,'FontSize',12,'FontName','Arial');
ylabel({'\fontname{Arial}Content (%)'},'FontSize',12,'LineWidth',1);
set(gca,'Ycolor','k');
axis([0.5,9.5,0,30]);
Leg1 = legend('\fontname{Arial}Chl a','\fontname{Arial}Chl b','\fontname{Arial}Carotenoids','FontSize',12);
set(Leg1,'Position',[0.2 0.74 0.15 0.15]);
ah=axes('position',get(gca,'position'),'visible','off');
Leg2 = legend(ah,[bw(4),bw(5)],'\fontname{Arial}GPT','\fontname{Arial}Amino Acid','FontSize',12);
set(Leg2,'Position',[0.72 0.74 0.15 0.15]);




