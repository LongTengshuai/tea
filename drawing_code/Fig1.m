clear all;
clc

hong_vnir = xlsread('D:\tea\数据\品种2\红柄可见.xlsx');
hong_swir = xlsread('D:\tea\数据\品种2\红柄短波.xlsx');
bai_vnir = xlsread('D:\tea\数据\品种2\白叶可见.xlsx');
bai_swir = xlsread('D:\tea\数据\品种2\白叶短波.xlsx');
feng_vnir = xlsread('D:\tea\数据\品种2\凤凰可见.xlsx');
feng_swir = xlsread('D:\tea\数据\品种2\凤凰短波.xlsx');

mean_h_vnir = mean(hong_vnir(2:44,16:161),1);
mean_h_swir = mean(hong_swir(2:44,21:239),1);
mean_b_vnir = mean(bai_vnir(2:35,16:161),1);
mean_b_swir = mean(bai_swir(2:35,21:239),1);
mean_f_vnir = mean(feng_vnir(2:60,16:161),1);
mean_f_swir = mean(feng_swir(2:60,21:239),1);

min_h_vnir = min(hong_vnir(2:44,16:161));
min_h_swir = min(hong_swir(2:44,21:239));
min_b_vnir = min(bai_vnir(2:35,16:161));
min_b_swir = min(bai_swir(2:35,21:239));
min_f_vnir = min(feng_vnir(2:60,16:161));
min_f_swir = min(feng_swir(2:60,21:239));

max_h_vnir = max(hong_vnir(2:44,16:161));
max_h_swir = max(hong_swir(2:44,21:239));
max_b_vnir = max(bai_vnir(2:35,16:161));
max_b_swir = max(bai_swir(2:35,21:239));
max_f_vnir = max(feng_vnir(2:60,16:161));
max_f_swir = max(feng_swir(2:60,21:239));

x1 = hong_vnir(1,16:161);
x2 = hong_swir(1,21:239);
x3 = hong_vnir(1,16:160);
x4 = hong_swir(1,21:238);

str1 = {'\fontname{Arial}1'};
str2 = {'\fontname{Arial}2'};
str3 = {'\fontname{Arial}3'};
str4 = {'\fontname{Arial}4'};

h = figure;
set(h,'position',[250 150 800 300]);
subplot(1,2,1)
plot(1:2,2:3,'k','LineWidth',4);hold on;
plot(1:2,2:3,'r','LineWidth',4);hold on;
plot(1:2,2:3,'y','LineWidth',4);hold on;
plot(x1,mean_h_vnir,'k','LineWidth',1);hold on;
plot(x1,mean_b_vnir,'r','LineWidth',1);hold on;
plot(x1,mean_f_vnir,'y','LineWidth',1);hold on;
fill([x1 fliplr(x1)],[min_h_vnir fliplr(max_h_vnir)],'k','FaceAlpha',0.8);
fill([x1 fliplr(x1)],[min_b_vnir fliplr(max_b_vnir)],'r','FaceAlpha',0.5);
fill([x1 fliplr(x1)],[min_f_vnir fliplr(max_f_vnir)],'y','FaceAlpha',0.5);
plot(x1,mean_h_vnir,'k','LineWidth',1.2);hold on;
plot(x1,mean_b_vnir,'r','LineWidth',1.2);hold on;
plot(x1,mean_f_vnir,'y','LineWidth',1.2);hold on;

plot(1:2,2:3,'k','LineWidth',4);hold on;
plot(1:2,2:3,'r','LineWidth',4);hold on;
plot(1:2,2:3,'y','LineWidth',4);hold on;
plot(x2,mean_h_swir,'k','LineWidth',1);hold on;
plot(x2,mean_b_swir,'r','LineWidth',1);hold on;
plot(x2,mean_f_swir,'y','LineWidth',1);hold on;
fill([x2 fliplr(x2)],[min_h_swir fliplr(max_h_swir)],'k','FaceAlpha',0.8);
fill([x2 fliplr(x2)],[min_b_swir fliplr(max_b_swir)],'r','FaceAlpha',0.5);
fill([x2 fliplr(x2)],[min_f_swir fliplr(max_f_swir)],'y','FaceAlpha',0.5);
plot(x2,mean_h_swir,'k','LineWidth',1.2);hold on;
plot(x2,mean_b_swir,'r','LineWidth',1.2);hold on;
plot(x2,mean_f_swir,'y','LineWidth',1.2);hold on;
% xlabel({'\fontname{Arial}Wavelength (nm)'},'FontSize',11,'LineWidth',1);
% ylabel({'\fontname{Arial}Reflectivity'},'FontSize',11,'LineWidth',1);
axis([400,1700,0,0.8]);
set(gca,'FontSize',12,'Fontname', 'Arial','LineWidth',1);
set(gca,'tickdir','out');
XL = get(gca,'xlim'); XR = XL(2);
YL = get(gca,'ylim'); YT = YL(2);
plot(XL,YT*ones(size(XL)),'color','k','LineWidth',1)
plot(XR*ones(size(YL)),YL,'color','k','LineWidth',1)
box off
% legend('\fontname{Arial}Hongbing','\fontname{Arial}Baiye','\fontname{Arial}Fenghuang','Location','southwest');

fd_hong_vnir=diff(hong_vnir(2:44,16:161)');fd_hong_vnir=fd_hong_vnir';
fd_hong_swir=diff(hong_swir(2:44,21:239)');fd_hong_swir=fd_hong_swir';
fd_bai_vnir=diff(bai_vnir(2:35,16:161)');fd_bai_vnir=fd_bai_vnir';
fd_bai_swir=diff(bai_swir(2:35,21:239)');fd_bai_swir=fd_bai_swir';
fd_feng_vnir=diff(feng_vnir(2:60,16:161)');fd_feng_vnir=fd_feng_vnir';
fd_feng_swir=diff(feng_swir(2:60,21:239)');fd_feng_swir=fd_feng_swir';

fd_mean_h_vnir = mean(fd_hong_vnir,1);
fd_mean_h_swir = mean(fd_hong_swir,1);
fd_mean_b_vnir = mean(fd_bai_vnir,1);
fd_mean_b_swir = mean(fd_bai_swir,1);
fd_mean_f_vnir = mean(fd_feng_vnir,1);
fd_mean_f_swir = mean(fd_feng_swir,1);

fd_min_h_vnir = min(fd_hong_vnir);
fd_min_h_swir = min(fd_hong_swir);
fd_min_b_vnir = min(fd_bai_vnir);
fd_min_b_swir = min(fd_bai_swir);
fd_min_f_vnir = min(fd_feng_vnir);
fd_min_f_swir = min(fd_feng_swir);

fd_max_h_vnir = max(fd_hong_vnir);
fd_max_h_swir = max(fd_hong_swir);
fd_max_b_vnir = max(fd_bai_vnir);
fd_max_b_swir = max(fd_bai_swir);
fd_max_f_vnir = max(fd_feng_vnir);
fd_max_f_swir = max(fd_feng_swir);

% h = figure;
% set(h,'position',[250 150 950 350]);
we = [x3(14) x3(14)];
we2 = [x3(48) x3(48)];
we3 = [x3(65) x3(65)];
we4 = [x3(107) x3(107)];
we5 = [-1 1];
subplot(1,2,2)
plot(1:2,2:3,'k','LineWidth',4);hold on;
plot(1:2,2:3,'r','LineWidth',4);hold on;
plot(1:2,2:3,'y','LineWidth',4);hold on;
fill([we we2],[we5 fliplr(we5)],'r','FaceAlpha',0.3);
fill([we3 we4],[we5 fliplr(we5)],'r','FaceAlpha',0.3);
plot(x3,fd_mean_h_vnir,'k','LineWidth',1);hold on;
plot(x3,fd_mean_b_vnir,'r','LineWidth',1);hold on;
plot(x3,fd_mean_f_vnir,'y','LineWidth',1);hold on;
fill([x3 fliplr(x3)],[fd_min_h_vnir fliplr(fd_max_h_vnir)],'k','FaceAlpha',0.8);
fill([x3 fliplr(x3)],[fd_min_b_vnir fliplr(fd_max_b_vnir)],'r','FaceAlpha',0.5);
fill([x3 fliplr(x3)],[fd_min_f_vnir fliplr(fd_max_f_vnir)],'y','FaceAlpha',0.5);
plot(x3,fd_mean_h_vnir,'k','LineWidth',1.2);hold on;
plot(x3,fd_mean_b_vnir,'r','LineWidth',1.2);hold on;
plot(x3,fd_mean_f_vnir,'y','LineWidth',1.2);hold on;

we1 = [x4(48) x4(48)];
we21 = [x4(78) x4(78)];
we31 = [x4(101) x4(101)];
we41 = [x4(208) x4(208)];
we51 = [-1 1];

plot(1:2,2:3,'k','LineWidth',4);hold on;
plot(1:2,2:3,'r','LineWidth',4);hold on;
plot(1:2,2:3,'y','LineWidth',4);hold on;
fill([we1 we21],[we51 fliplr(we51)],'r','FaceAlpha',0.3);
fill([we31 we41],[we51 fliplr(we51)],'r','FaceAlpha',0.3);
plot(x4,fd_mean_h_swir,'k','LineWidth',1);hold on;
plot(x4,fd_mean_b_swir,'r','LineWidth',1);hold on;
plot(x4,fd_mean_f_swir,'y','LineWidth',1);hold on;
fill([x4 fliplr(x4)],[fd_min_h_swir fliplr(fd_max_h_swir)],'k','FaceAlpha',0.8);
fill([x4 fliplr(x4)],[fd_min_b_swir fliplr(fd_max_b_swir)],'r','FaceAlpha',0.5);
fill([x4 fliplr(x4)],[fd_min_f_swir fliplr(fd_max_f_swir)],'y','FaceAlpha',0.5);
plot(x4,fd_mean_h_swir,'k','LineWidth',1.2);hold on;
plot(x4,fd_mean_b_swir,'r','LineWidth',1.2);hold on;
plot(x4,fd_mean_f_swir,'y','LineWidth',1.2);hold on;
% xlabel({'\fontname{Arial}Wavelength (nm)'},'FontSize',11,'LineWidth',1);
% ylabel({'\fontname{Arial}Reflectivity'},'FontSize',11,'LineWidth',1);
axis([400,1700,-0.03,0.04]);
set(gca,'FontSize',12,'Fontname', 'Arial','LineWidth',1);
set(gca,'tickdir','out');
text(540,-0.022,str1,'FontSize',12);
text(725,-0.022,str2,'FontSize',12)
text(1140,-0.022,str3,'FontSize',12);
text(1370,-0.022,str4,'FontSize',12)
XL = get(gca,'xlim'); XR = XL(2);
YL = get(gca,'ylim'); YT = YL(2);
plot(XL,YT*ones(size(XL)),'color','k','LineWidth',1)
plot(XR*ones(size(YL)),YL,'color','k','LineWidth',1)
box off
% legend('\fontname{Arial}Hongbing','\fontname{Arial}Baiye','\fontname{Arial}Fenghuang','\fontname{Arial}Spectral interval','Location','southwest');
