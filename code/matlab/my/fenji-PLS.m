%高光谱光谱测定TVB-N数据处理2011.11.29
% A=[];
% for i=1:18
% s=strcat('J:\肉样检测\预处理\提取特征图像11.27\第1天\',num2str(i),'.txt');
% t=importdata(s);f=t.data;
% k=f(:,:);
% A=[A k];
% end
% save('J:\肉样检测\预处理\提取特征图像11.27\第1天\A.xls','A','-ascii');%保存为A.xls文件
% %先将(J:\肉样检测\预处理\提取特征图像11.27\第1天)文件夹内各原始光谱的CK1.txt、CK2.txt等对应改为罗马数字序号1.txt、2.txt等，以便读取进入A中。
% 
% x2=[];
% for i=1:18
% s=strcat('J:\肉样检测\预处理\提取特征图像11.27\第3天\',num2str(i),'.txt');
% t=importdata(s);f=t.data;
% k=f(:,:);
% x2=[x2 k];
% end
% x3=[];
% for i=1:18
% s=strcat('J:\肉样检测\预处理\提取特征图像11.27\第5天\',num2str(i),'.txt');
% t=importdata(s);f=t.data;
% k=f(:,:);
% x3=[x3 k];
% end
% x5=[];
% for i=1:18
% s=strcat('J:\肉样检测\预处理\提取特征图像11.27\第5天\',num2str(i),'.txt');
% t=importdata(s);f=t.data;
% k=f(:,:);
% x5=[x5 k];
% end
% x7=[];
% for i=1:18
% s=strcat('J:\肉样检测\预处理\提取特征图像11.27\第5天\',num2str(i),'.txt');
% t=importdata(s);f=t.data;
% k=f(:,:);
% x7=[x7 k];
% end
% x9=[];
% for i=1:18
% s=strcat('J:\肉样检测\预处理\提取特征图像11.27\第5天\',num2str(i),'.txt');
% t=importdata(s);f=t.data;
% k=f(:,:);
% x9=[x9 k];
% end
% x11=[];
% for i=1:18
% s=strcat('J:\肉样检测\预处理\提取特征图像11.27\第5天\',num2str(i),'.txt');
% t=importdata(s);f=t.data;
% k=f(:,:);
% x11=[x11 k];
% end
% save('J:\肉样检测\预处理\提取特征图像11.27\第1天\x1.xls','x1','-ascii');
% 
% x=x1(1,:);
% x=x1'(1,:);
% x01=x1';
% x=x01(1,:);
% I3=x3(1,:)';
% I3=x3'(1,:);
% x3=x3';
% x5=x5';
% x7=x7';
% x9=x79';
% x9=x9';
% x11=x11';
% x1=x1';
% clear
% clc
% 
% X=[x1;x3;x5;x7;x9;x11];%X为变量值汇总得到X108*618
% Xq=X(:,1:5:618);%从X中每5个取1个变量得到XX108*124
% whos Xq
% x=xo(1:5:618);%从xo(为最初波长值xo618*1)中每5个取1个变量（与X对应）得到x124*1
% plot(x,Xq)
% %-- 11-11-29  下午12:33 --%
% [Xsm5] = smooth(XX,5);%对XX取5（5为平滑窗口从光谱原始数据取的点数，以这5个点拟合得到回归曲线，算得中心点的值并取代，然后平滑窗口逐点移动进行拟合计算和取代，5为可改参数，但只能取奇数，其取值越大，平滑拟合计算时取点越多。）
% Xsm5r=reshape(Xsm5,[108 124]);%XX平滑后得到的smdata是一条直线，需要重排为XX同样的变量矩阵
% whos Xsm5r
% figure (2)%打开绘图窗口（不写则覆盖figure1中的图片！）
% whos x
% plot(x,Xsm5r);
% [Xsm9] = smooth(Xqs,9);%对Xqs取9点平滑
% Xsm9r=reshape(Xsm9,[108 124]);
% figure (3)
% plot(x,Xsm9r);
% Xqs=Xsm9r;
% xqssnv=zscore(Xqs');%此句同xqssnv=SNV(Xqs');即进行SNV预处理
% xqssnv=xqssnv';
% whos xqssnv
% figure (5)
% plot(x,xqssnv);
% 
% [Ystvbn z2]=sort(Ytvbn);%对Ytvbn由小到大排序得到Ystvbn
% Xs=X(z2,:);%与Ystvbn的变量X也由小到大排序得到Xs
% 
% 
% y1=[Ystvbn(1:3:108)];
% y2=[Ystvbn(2:3:108)];
% y3=[Ystvbn(3:3:108)];
% Ytr=[y1;y3];
% Yte=y2;
% x1=[Xs(1:3:108,:)];
% x2=[Xs(2:3:108,:)];
% x3=[Xs(3:3:108,:)];
% Xtr=[x1;x3];
% Xte=x2;
% %对排序后的Ystvbn和Xs分训练集和预测值
% 
% xsnv=SNV(Xtr);
% xtnv=SNV(Xte);
% d1Xtr=deriv(Xtr,1);
% d1Xte=deriv(Xte,1);
% d2Xtr=deriv(Xtr,2);
% d2Xte=deriv(Xte,2);
% [xmsc,me]=MSC(Xtr,1,124);
% [xtmsc,me]=MSC(Xte,1,124);
% [xcen,me,ctest]=center(Xtr,1,Xtr);
% [xtcen,me,ctest]=center(Xte,1,Xte);
% %对变量Xtr、Xte进行SNV、一阶导数、二阶导数、MSC和中心化预处理

% %ipls方法
% Model=ipls(xsnv,Ytr,15,'mean',20,x,'syst123',5);%注意改路径 E:\燕华\数据处理全部相关内容\itoolbox 定量路径,具体每个参数的意义可以参照笔记
% plsrmse(Model,0);%得出主成分数，0是个参数
% iplsplot(Model,'intlabel'); %画图比较RMSECV
% iplsplot(Model,'wavlabel');%横坐标改变为波长
% plspvsm(Model,4,11);%用第11个区间的4个主成分画图
% oneModel=plsmodel(Model,11,10,'mean','test',5);%syst123 改为test 好像效果区别不大
% predModel=plspredict(xtnv,oneModel,4,Yte);
% plspvsm(predModel,4);
% predModel.Ypred(:,:,3);%这步是看结果，结构体看结果，但是似乎意义不大。

%PLS方法
Model=ipls(xsnv,Ytr,10,'mean',1,x,'syst123',5);%20个区间改为1个区间，就是PLS方法
plsrmse(Model,0);%得出主成分数，0是个参数
plspvsm(Model,5,1);%1个区间的5个主成分画图，是上一步得出的 因为是全局变量 所以只有1个区间
oneModel=plsmodel(Model,1,10,'mean','test',5);
predModel=plspredict(xtnv,oneModel,5,Yte);
plspvsm(predModel,5);%预测集的结果


% %siPLS方法
% siModel=sipls(xsnv,Ytr,10,'mscmean',10,2,x,'syst123',5);%后10为区间数，一般从10试到30，如果30是最好，还要往后试，3是联合区间数，取值2,3,4，
%                                                           %每个区间都要变，所以要训练30*3=120次,x是横坐标，默认值用[]
% siModel=sipls(xsnv,Ytr,10,'mscmean',10,3,x,'syst123',5);%以此类推
% siplstable(siModel);%显示训练结果，第一行是最好的，所以最后要用Excel统计结果
% intervals(siModel);%看Model的详细信息，有哪些变量 波长起始范围
% FinalModel=plsmodel(siModel,[4 7 13],7,'mscmean','syst123',5);%把上面训练好的结果最为最后的模型
% plsrmse(FinalModel);%画图看结果
% plspvsm(FinalModel,7);%7是最佳主成分数
% oneModel=plsmodel(siModel,[4 7 13],7,'mscmean','test',5);
% predModel=plspredict(xtnv,oneModel,7,Yte);
% plspvsm(predModel,7);

