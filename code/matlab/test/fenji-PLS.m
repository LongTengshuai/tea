%�߹��׹��ײⶨTVB-N���ݴ���2011.11.29
% A=[];
% for i=1:18
% s=strcat('J:\�������\Ԥ����\��ȡ����ͼ��11.27\��1��\',num2str(i),'.txt');
% t=importdata(s);f=t.data;
% k=f(:,:);
% A=[A k];
% end
% save('J:\�������\Ԥ����\��ȡ����ͼ��11.27\��1��\A.xls','A','-ascii');%����ΪA.xls�ļ�
% %�Ƚ�(J:\�������\Ԥ����\��ȡ����ͼ��11.27\��1��)�ļ����ڸ�ԭʼ���׵�CK1.txt��CK2.txt�ȶ�Ӧ��Ϊ�����������1.txt��2.txt�ȣ��Ա��ȡ����A�С�
% 
% x2=[];
% for i=1:18
% s=strcat('J:\�������\Ԥ����\��ȡ����ͼ��11.27\��3��\',num2str(i),'.txt');
% t=importdata(s);f=t.data;
% k=f(:,:);
% x2=[x2 k];
% end
% x3=[];
% for i=1:18
% s=strcat('J:\�������\Ԥ����\��ȡ����ͼ��11.27\��5��\',num2str(i),'.txt');
% t=importdata(s);f=t.data;
% k=f(:,:);
% x3=[x3 k];
% end
% x5=[];
% for i=1:18
% s=strcat('J:\�������\Ԥ����\��ȡ����ͼ��11.27\��5��\',num2str(i),'.txt');
% t=importdata(s);f=t.data;
% k=f(:,:);
% x5=[x5 k];
% end
% x7=[];
% for i=1:18
% s=strcat('J:\�������\Ԥ����\��ȡ����ͼ��11.27\��5��\',num2str(i),'.txt');
% t=importdata(s);f=t.data;
% k=f(:,:);
% x7=[x7 k];
% end
% x9=[];
% for i=1:18
% s=strcat('J:\�������\Ԥ����\��ȡ����ͼ��11.27\��5��\',num2str(i),'.txt');
% t=importdata(s);f=t.data;
% k=f(:,:);
% x9=[x9 k];
% end
% x11=[];
% for i=1:18
% s=strcat('J:\�������\Ԥ����\��ȡ����ͼ��11.27\��5��\',num2str(i),'.txt');
% t=importdata(s);f=t.data;
% k=f(:,:);
% x11=[x11 k];
% end
% save('J:\�������\Ԥ����\��ȡ����ͼ��11.27\��1��\x1.xls','x1','-ascii');
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
% X=[x1;x3;x5;x7;x9;x11];%XΪ����ֵ���ܵõ�X108*618
% Xq=X(:,1:5:618);%��X��ÿ5��ȡ1�������õ�XX108*124
% whos Xq
% x=xo(1:5:618);%��xo(Ϊ�������ֵxo618*1)��ÿ5��ȡ1����������X��Ӧ���õ�x124*1
% plot(x,Xq)
% %-- 11-11-29  ����12:33 --%
% [Xsm5] = smooth(XX,5);%��XXȡ5��5Ϊƽ�����ڴӹ���ԭʼ����ȡ�ĵ���������5������ϵõ��ع����ߣ�������ĵ��ֵ��ȡ����Ȼ��ƽ����������ƶ�������ϼ����ȡ����5Ϊ�ɸĲ�������ֻ��ȡ��������ȡֵԽ��ƽ����ϼ���ʱȡ��Խ�ࡣ��
% Xsm5r=reshape(Xsm5,[108 124]);%XXƽ����õ���smdata��һ��ֱ�ߣ���Ҫ����ΪXXͬ���ı�������
% whos Xsm5r
% figure (2)%�򿪻�ͼ���ڣ���д�򸲸�figure1�е�ͼƬ����
% whos x
% plot(x,Xsm5r);
% [Xsm9] = smooth(Xqs,9);%��Xqsȡ9��ƽ��
% Xsm9r=reshape(Xsm9,[108 124]);
% figure (3)
% plot(x,Xsm9r);
% Xqs=Xsm9r;
% xqssnv=zscore(Xqs');%�˾�ͬxqssnv=SNV(Xqs');������SNVԤ����
% xqssnv=xqssnv';
% whos xqssnv
% figure (5)
% plot(x,xqssnv);
% 
% [Ystvbn z2]=sort(Ytvbn);%��Ytvbn��С��������õ�Ystvbn
% Xs=X(z2,:);%��Ystvbn�ı���XҲ��С��������õ�Xs
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
% %��������Ystvbn��Xs��ѵ������Ԥ��ֵ
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
% %�Ա���Xtr��Xte����SNV��һ�׵��������׵�����MSC�����Ļ�Ԥ����

% %ipls����
% Model=ipls(xsnv,Ytr,15,'mean',20,x,'syst123',5);%ע���·�� E:\�໪\���ݴ���ȫ���������\itoolbox ����·��,����ÿ��������������Բ��ձʼ�
% plsrmse(Model,0);%�ó����ɷ�����0�Ǹ�����
% iplsplot(Model,'intlabel'); %��ͼ�Ƚ�RMSECV
% iplsplot(Model,'wavlabel');%������ı�Ϊ����
% plspvsm(Model,4,11);%�õ�11�������4�����ɷֻ�ͼ
% oneModel=plsmodel(Model,11,10,'mean','test',5);%syst123 ��Ϊtest ����Ч�����𲻴�
% predModel=plspredict(xtnv,oneModel,4,Yte);
% plspvsm(predModel,4);
% predModel.Ypred(:,:,3);%�ⲽ�ǿ�������ṹ�忴����������ƺ����岻��

%PLS����
Model=ipls(xsnv,Ytr,10,'mean',1,x,'syst123',5);%20�������Ϊ1�����䣬����PLS����
plsrmse(Model,0);%�ó����ɷ�����0�Ǹ�����
plspvsm(Model,5,1);%1�������5�����ɷֻ�ͼ������һ���ó��� ��Ϊ��ȫ�ֱ��� ����ֻ��1������
oneModel=plsmodel(Model,1,10,'mean','test',5);
predModel=plspredict(xtnv,oneModel,5,Yte);
plspvsm(predModel,5);%Ԥ�⼯�Ľ��


% %siPLS����
% siModel=sipls(xsnv,Ytr,10,'mscmean',10,2,x,'syst123',5);%��10Ϊ��������һ���10�Ե�30�����30����ã���Ҫ�����ԣ�3��������������ȡֵ2,3,4��
%                                                           %ÿ�����䶼Ҫ�䣬����Ҫѵ��30*3=120��,x�Ǻ����꣬Ĭ��ֵ��[]
% siModel=sipls(xsnv,Ytr,10,'mscmean',10,3,x,'syst123',5);%�Դ�����
% siplstable(siModel);%��ʾѵ���������һ������õģ��������Ҫ��Excelͳ�ƽ��
% intervals(siModel);%��Model����ϸ��Ϣ������Щ���� ������ʼ��Χ
% FinalModel=plsmodel(siModel,[4 7 13],7,'mscmean','syst123',5);%������ѵ���õĽ����Ϊ����ģ��
% plsrmse(FinalModel);%��ͼ�����
% plspvsm(FinalModel,7);%7��������ɷ���
% oneModel=plsmodel(siModel,[4 7 13],7,'mscmean','test',5);
% predModel=plspredict(xtnv,oneModel,7,Yte);
% plspvsm(predModel,7);

