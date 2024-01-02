clear all;
clc
rt = imread('C:\Users\longteng\Desktop\新建文件夹\5\H1.tif');
rty1 = rt(111:1101,:,3)/255;
rt2 = imread('C:\Users\longteng\Desktop\新建文件夹\5\H2.tif');
rty2 = rt2(111:1101,:,3)/255;
rt3 = imread('C:\Users\longteng\Desktop\新建文件夹\5\B1.tif');
rty3 = rt3(111:1101,:,3)/255;
rt4 = imread('C:\Users\longteng\Desktop\新建文件夹\5\B2.tif');
rty4 = rt4(111:1101,:,3)/255;
rt5 = imread('C:\Users\longteng\Desktop\新建文件夹\5\B3.tif');
rty5 = rt5(111:1101,:,3)/255;
rt6 = imread('C:\Users\longteng\Desktop\新建文件夹\5\F1.tif');
rty6 = rt6(111:1101,:,3)/255;
rt7 = imread('C:\Users\longteng\Desktop\新建文件夹\5\F2.tif');
rty7 = rt7(111:1101,:,3)/255;
rt8 = imread('C:\Users\longteng\Desktop\新建文件夹\5\F3.tif');
rty8 = rt8(111:1101,:,3)/255;
rt9 = imread('C:\Users\longteng\Desktop\新建文件夹\5\H3.tif');
rty9 = rt9(111:1101,:,3)/255;



H2 = imread('C:\Users\longteng\Desktop\新建文件夹\5\H1.bmp');
H3 = imread('C:\Users\longteng\Desktop\新建文件夹\5\H2.bmp');
H4 = imread('C:\Users\longteng\Desktop\新建文件夹\5\H3.bmp');
B2 = imread('C:\Users\longteng\Desktop\新建文件夹\5\B1.bmp');
B3 = imread('C:\Users\longteng\Desktop\新建文件夹\5\B2.bmp');
B4 = imread('C:\Users\longteng\Desktop\新建文件夹\5\B3.bmp');
F2 = imread('C:\Users\longteng\Desktop\新建文件夹\5\F1.bmp');
F3 = imread('C:\Users\longteng\Desktop\新建文件夹\5\F2.bmp');
F4 = imread('C:\Users\longteng\Desktop\新建文件夹\5\F3.bmp');

H2_1 = H2(111:1101,:,:);
H3_1 = H3(111:1101,:,:);
H4_1 = H4(111:1101,:,:);
B2_1 = B2(111:1101,:,:);
B3_1 = B3(111:1101,:,:);
B4_1 = B4(111:1101,:,:);
F2_1 = F2(111:1101,:,:);
F3_1 = F3(111:1101,:,:);
F4_1 = F4(111:1101,:,:);

H22 = imread('C:\Users\longteng\Desktop\新建文件夹\3\H1\786.8.bmp');
H32 = imread('C:\Users\longteng\Desktop\新建文件夹\3\H2\786.8.bmp');
H42 = imread('C:\Users\longteng\Desktop\新建文件夹\3\H3\786.8.bmp');
B22 = imread('C:\Users\longteng\Desktop\新建文件夹\3\B1\786.8.bmp');
B32 = imread('C:\Users\longteng\Desktop\新建文件夹\3\B2\786.8.bmp');
B42 = imread('C:\Users\longteng\Desktop\新建文件夹\3\B3\786.8.bmp');
F22 = imread('C:\Users\longteng\Desktop\新建文件夹\3\F1\786.8.bmp');
F32 = imread('C:\Users\longteng\Desktop\新建文件夹\3\F2\786.8.bmp');
F42 = imread('C:\Users\longteng\Desktop\新建文件夹\3\F3\786.8.bmp');
 
H2_2 = H22(111:1101,:,:);
H3_2 = H32(111:1101,:,:);
H4_2 = H42(111:1101,:,:);
B2_2 = B22(111:1101,:,:);
B3_2 = B32(111:1101,:,:);
B4_2 = B42(111:1101,:,:);
F2_2 = F22(111:1101,:,:);
F3_2 = F32(111:1101,:,:);
F4_2 = F42(111:1101,:,:);

for i = 1:991
    for j = 1:960
        if rty1(i,j) == 1
            er = H2_2(i,j);
            H2_2(i,j) = 0;
        end
        if rty2(i,j) == 1
            er = H3_2(i,j);
            H3_2(i,j) = 0;
        end      
        if rty9(i,j) == 1
            er = H4_2(i,j);
            H4_2(i,j) = 0;
        end 
        if H4_2(i,j) >= 110
            er = H4_2(i,j);
            H4_2(i,j) = er-18;
        end 
        
        if rty3(i,j) == 1
            er = B2_2(i,j);
            B2_2(i,j) = 0;
        end
        if rty4(i,j) == 1
            er = B3_2(i,j);
            B3_2(i,j) = 0;
        end        
        if rty5(i,j) == 1
            er = B4_2(i,j);
            B4_2(i,j) = 0;
        end        
        if rty6(i,j) == 1
            er = F2_2(i,j);
            F2_2(i,j) = 0;
        end  
        if rty7(i,j) == 1
            er = F3_2(i,j);
            F3_2(i,j) = 0;
        end  

        if rty8(i,j) == 1
            er = F4_2(i,j);
            F4_2(i,j) = 0;
        end       
    end
end
for p = 1:991
    for t = 1:960
        if F2_2(p,t) >= 120
            er = F2_2(p,t);
            F2_2(p,t) = er-20;
        end   
        
    end
end

for p = 1:991
    for t = 1:450
        if F3_2(p,t) >= 120
            er = F3_2(p,t);
            F3_2(p,t) = er-20;
        end   
        
    end
end


H23 = imread('C:\Users\longteng\Desktop\新建文件夹\3\H1\851.1.bmp');
H33 = imread('C:\Users\longteng\Desktop\新建文件夹\3\H2\851.1.bmp');
H43 = imread('C:\Users\longteng\Desktop\新建文件夹\3\H3\851.1.bmp');
B23 = imread('C:\Users\longteng\Desktop\新建文件夹\3\B1\851.1.bmp');
B33 = imread('C:\Users\longteng\Desktop\新建文件夹\3\B2\851.1.bmp');
B43 = imread('C:\Users\longteng\Desktop\新建文件夹\3\B3\851.1.bmp');
F23 = imread('C:\Users\longteng\Desktop\新建文件夹\3\F1\851.1.bmp');
F33 = imread('C:\Users\longteng\Desktop\新建文件夹\3\F2\851.1.bmp');
F43 = imread('C:\Users\longteng\Desktop\新建文件夹\3\F3\851.1.bmp');

H2_3 = H23(111:1101,:,:);
H3_3 = H33(111:1101,:,:);
H4_3 = H43(111:1101,:,:);
B2_3 = B23(111:1101,:,:);
B3_3 = B33(111:1101,:,:);
B4_3 = B43(111:1101,:,:);
F2_3 = F23(111:1101,:,:);
F3_3 = F33(111:1101,:,:);
F4_3 = F43(111:1101,:,:);

for i = 1:991
    for j = 1:960
        if rty1(i,j) == 1
            er = H2_3(i,j);
            H2_3(i,j) = 0;
        end
        if rty2(i,j) == 1
            er = H3_3(i,j);
            H3_3(i,j) = 0;
        end      
        if rty9(i,j) == 1
            er = H4_3(i,j);
            H4_3(i,j) = 0;
        end        
        if H4_3(i,j) >= 60
            er = H4_3(i,j);
            H4_3(i,j) = er-8;
        end 
        
        if rty3(i,j) == 1
            er = B2_3(i,j);
            B2_3(i,j) = 0;
        end
        if rty4(i,j) == 1
            er = B3_3(i,j);
            B3_3(i,j) = 0;
        end        
        if rty5(i,j) == 1
            er = B4_3(i,j);
            B4_3(i,j) = 0;
        end        
        if rty6(i,j) == 1
            er = F2_3(i,j);
            F2_3(i,j) = 0;
        end  
        if rty7(i,j) == 1
            er = F3_3(i,j);
            F3_3(i,j) = 0;
        end   
        if rty8(i,j) == 1
            er = F4_3(i,j);
            F4_3(i,j) = 0;
        end       
    end
end

for p = 1:991
    for t = 1:960
        if F2_3(p,t) >= 120
            er = F2_3(p,t);
            F2_3(p,t) = er-20;
        end   
        
    end
end
for p = 1:991
    for t = 1:450
        if F3_3(p,t) >= 70
            er = F3_3(p,t);
            F3_3(p,t) = er-8;
        end   
        
    end
end

H24 = imread('C:\Users\longteng\Desktop\新建文件夹\3\H1\833.2.bmp');
H34 = imread('C:\Users\longteng\Desktop\新建文件夹\3\H2\833.2.bmp');
H44 = imread('C:\Users\longteng\Desktop\新建文件夹\3\H3\772.7.bmp');
B24 = imread('C:\Users\longteng\Desktop\新建文件夹\3\B1\833.2.bmp');
B34 = imread('C:\Users\longteng\Desktop\新建文件夹\3\B2\833.2.bmp');
B44 = imread('C:\Users\longteng\Desktop\新建文件夹\3\B3\833.2.bmp');
F24 = imread('C:\Users\longteng\Desktop\新建文件夹\3\F1\833.2.bmp');
F34 = imread('C:\Users\longteng\Desktop\新建文件夹\3\F2\833.2.bmp');
F44 = imread('C:\Users\longteng\Desktop\新建文件夹\3\F3\833.2.bmp');

H2_4 = H24(111:1101,:,:);
H3_4 = H34(111:1101,:,:);
H4_4 = H44(111:1101,:,:);
B2_4 = B24(111:1101,:,:);
B3_4 = B34(111:1101,:,:);
B4_4 = B44(111:1101,:,:);
F2_4 = F24(111:1101,:,:);
F3_4 = F34(111:1101,:,:);
F4_4 = F44(111:1101,:,:);

for i = 1:991
    for j = 1:960
        if rty1(i,j) == 1
            er = H2_4(i,j);
            H2_4(i,j) = 0;
        end
        if rty2(i,j) == 1
            er = H3_4(i,j);
            H3_4(i,j) = 0;
        end      
        if rty9(i,j) == 1
            er = H4_4(i,j);
            H4_4(i,j) = 0;
        end
        if H4_4(i,j) >= 120
            er = H4_4(i,j);
            H4_4(i,j) = er-20;
        end 
        
        if rty3(i,j) == 1
            er = B2_4(i,j);
            B2_4(i,j) = 0;
        end
        if rty4(i,j) == 1
            er = B3_4(i,j);
            B3_4(i,j) = 0;
        end        
        if rty5(i,j) == 1
            er = B4_4(i,j);
            B4_4(i,j) = 0;
        end        
        if rty6(i,j) == 1
            er = F2_4(i,j);
            F2_4(i,j) = 0;
        end  
        if rty7(i,j) == 1
            er = F3_4(i,j);
            F3_4(i,j) = 0;
        end   
        if rty8(i,j) == 1
            er = F4_4(i,j);
            F4_4(i,j) = 0;
        end       
    end
end

for p = 1:991
    for t = 1:960
        if F2_4(p,t) >= 110
            er = F2_4(p,t);
            F2_4(p,t) = er-20;
        end   
        
    end
end

for p = 1:991
    for t = 1:450
        if F3_4(p,t) >= 80
            er = F3_4(p,t);
            F3_4(p,t) = er-12;
        end   
        
    end
end


H25 = imread('C:\Users\longteng\Desktop\新建文件夹\3\H1\811.7.bmp');
H35 = imread('C:\Users\longteng\Desktop\新建文件夹\3\H2\811.7.bmp');
H45 = imread('C:\Users\longteng\Desktop\新建文件夹\3\H3\811.7.bmp');
B25 = imread('C:\Users\longteng\Desktop\新建文件夹\3\B1\811.7.bmp');
B35 = imread('C:\Users\longteng\Desktop\新建文件夹\3\B2\811.7.bmp');
B45 = imread('C:\Users\longteng\Desktop\新建文件夹\3\B3\811.7.bmp');
F25 = imread('C:\Users\longteng\Desktop\新建文件夹\3\F1\811.7.bmp');
F35 = imread('C:\Users\longteng\Desktop\新建文件夹\3\F2\811.7.bmp');
F45 = imread('C:\Users\longteng\Desktop\新建文件夹\3\F3\811.7.bmp');

H2_5 = H25(111:1101,:,:);
H3_5 = H35(111:1101,:,:);
H4_5 = H45(111:1101,:,:);
B2_5 = B25(111:1101,:,:);
B3_5 = B35(111:1101,:,:);
B4_5 = B45(111:1101,:,:);
F2_5 = F25(111:1101,:,:);
F3_5 = F35(111:1101,:,:);
F4_5 = F45(111:1101,:,:);

for i = 1:991
    for j = 1:960
        if rty1(i,j) == 1
            er = H2_5(i,j);
            H2_5(i,j) = 0;
        end
        if rty2(i,j) == 1
            er = H3_5(i,j);
            H3_5(i,j) = 0;
        end      
        if rty9(i,j) == 1
            er = H4_5(i,j);
            H4_5(i,j) = 0;
        end  
        if H4_5(i,j) >= 90
            er = H4_5(i,j);
            H4_5(i,j) = er-18;
        end      
        
        if rty3(i,j) == 1
            er = B2_5(i,j);
            B2_5(i,j) = 0;
        end
        if rty4(i,j) == 1
            er = B3_5(i,j);
            B3_5(i,j) = 0;
        end        
        if rty5(i,j) == 1
            er = B4_5(i,j);
            B4_5(i,j) = 0;
        end        
        if rty6(i,j) == 1
            er = F2_5(i,j);
            F2_5(i,j) = 0;
        end  
        if rty7(i,j) == 1
            er = F3_5(i,j);
            F3_5(i,j) = 0;
        end   
        if rty8(i,j) == 1
            er = F4_5(i,j);
            F4_5(i,j) = 0;
        end       
    end
end

for p = 1:991
    for t = 1:450
        if F2_5(p,t) >= 105
            er = F2_5(p,t);
            F2_5(p,t) = er-20;
        end   
    end
end

for p = 1:991
    for t = 1:450
        if F3_5(p,t) >= 120
            er = F3_5(p,t);
            F3_5(p,t) = er-30;
        end   
        
    end
end

H26 = imread('C:\Users\longteng\Desktop\新建文件夹\3\H1\843.9.bmp');
H36 = imread('C:\Users\longteng\Desktop\新建文件夹\3\H2\843.9.bmp');
H46 = imread('C:\Users\longteng\Desktop\新建文件夹\3\H3\843.9.bmp');
B26 = imread('C:\Users\longteng\Desktop\新建文件夹\3\B1\843.9.bmp');
B36 = imread('C:\Users\longteng\Desktop\新建文件夹\3\B2\843.9.bmp');
B46 = imread('C:\Users\longteng\Desktop\新建文件夹\3\B3\843.9.bmp');
F26 = imread('C:\Users\longteng\Desktop\新建文件夹\3\F1\843.9.bmp');
F36 = imread('C:\Users\longteng\Desktop\新建文件夹\3\F2\843.9.bmp');
F46 = imread('C:\Users\longteng\Desktop\新建文件夹\3\F3\843.9.bmp');

H2_6 = H26(111:1101,:,:);
H3_6 = H36(111:1101,:,:);
H4_6 = H46(111:1101,:,:);
B2_6 = B26(111:1101,:,:);
B3_6 = B36(111:1101,:,:);
B4_6 = B46(111:1101,:,:);
F2_6 = F26(111:1101,:,:);
F3_6 = F36(111:1101,:,:);
F4_6 = F46(111:1101,:,:);
for i = 1:991
    for j = 1:960
        if rty1(i,j) == 1
            er = H2_6(i,j);
            H2_6(i,j) = 0;
        end
        if rty2(i,j) == 1
            er = H3_6(i,j);
            H3_6(i,j) = 0;
        end      
        if rty9(i,j) == 1
            er = H4_6(i,j);
            H4_6(i,j) = 0;
        end   
        if H4_6(i,j) >= 70
            er = H4_6(i,j);
            H4_6(i,j) = er-10;
        end          
        if rty3(i,j) == 1
            er = B2_6(i,j);
            B2_6(i,j) = 0;
        end
        if rty4(i,j) == 1
            er = B3_6(i,j);
            B3_6(i,j) = 0;
        end        
        if rty5(i,j) == 1
            er = B4_6(i,j);
            B4_6(i,j) = 0;
        end        
        if rty6(i,j) == 1
            er = F2_6(i,j);
            F2_6(i,j) = 0;
        end  
        if rty7(i,j) == 1
            er = F3_6(i,j);
            F3_6(i,j) = 0;
        end   
        if rty8(i,j) == 1
            er = F4_6(i,j);
            F4_6(i,j) = 0;
        end       
    end
end
for p = 1:991
    for t = 1:450
        if F2_6(p,t) >= 70
            er = F2_6(p,t);
            F2_6(p,t) = er-10;
        end   
    end
end

h = figure;
set(h,'position',[250 50 1450 950]);
subplot(6,9,1)
imshow(H2_1);
subplot(6,9,2)
imshow(H3_1);
subplot(6,9,3)
imshow(H4_1);
subplot(6,9,4)
imshow(B2_1);
subplot(6,9,5)
imshow(B3_1);
subplot(6,9,6)
imshow(B4_1);
subplot(6,9,7)
imshow(F2_1);
subplot(6,9,8)
imshow(F3_1);
subplot(6,9,9)
imshow(F4_1);


we1 = max(max(H2_2));
we2 = max(max(H3_2));
we3 = max(max(H4_2));
we4 = max(max(B2_2));
we5 = max(max(B3_2));
we6 = max(max(B4_2));
we7 = max(max(F2_2));
we8 = max(max(F3_2));
we9 = max(max(F4_2));


A = [45025/120,45025/192,45025/318,45025/185,45025/175,45025/224,45025/161,45025/251,45025/222];

subplot(6,9,10)
clim=[0 A(1)];
imTest1 = imagesc(H2_2,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
subplot(6,9,11)
clim=[0 A(2)];
imTest2 = imagesc(H3_2,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
subplot(6,9,12)
clim=[0 A(3)];
imTest3 = imagesc(H4_2,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
subplot(6,9,13)
clim=[0 A(4)];
imTest4 = imagesc(B2_2,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
subplot(6,9,14)
clim=[0 A(5)];
imTest5 = imagesc(B3_2,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
subplot(6,9,15)
clim=[0 A(6)];
imTest6 = imagesc(B4_2,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
subplot(6,9,16)
clim=[0 A(7)];
imTest7 = imagesc(F2_2,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
subplot(6,9,17)
clim=[0 A(8)];
imTest8 = imagesc(F3_2,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
subplot(6,9,18)
clim=[0 A(9)];
imTest9 = imagesc(F4_2,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
colormap jet;

ER = 7000;
B = [ER/35,ER/52,ER/93,ER/54,ER/41,ER/64,ER/42,6500/64,ER/62];
subplot(6,9,19)
clim=[0 B(1)];
imTest10 = imagesc(H2_3,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
subplot(6,9,20)
clim=[0 B(2)];
imTest11 = imagesc(H3_3,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
subplot(6,9,21)
clim=[0 B(3)];
imTest12 = imagesc(H4_3,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
subplot(6,9,22)
clim=[0 B(4)];
imTest13 = imagesc(B2_3,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
subplot(6,9,23)
clim=[0 B(5)];
imTest14 = imagesc(B3_3,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
subplot(6,9,24)
clim=[0 B(6)];
imTest15 = imagesc(B4_3,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
subplot(6,9,25)
clim=[0 B(7)];
imTest16 = imagesc(F2_3,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
subplot(6,9,26)
clim=[0 B(8)];
imTest17 = imagesc(F3_3,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
subplot(6,9,27)
clim=[0 B(9)];
imTest18 = imagesc(F4_3,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
colormap jet;

ER = 10000;
C = [ER/42,ER/68,17000/109,ER/67,ER/59,ER/81,ER/55,ER/84,ER/75];
subplot(6,9,28)
clim=[0 C(1)];
imTest19 = imagesc(H2_4,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
subplot(6,9,29)
clim=[0 C(2)];
imTest20 = imagesc(H3_4,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
subplot(6,9,30)
clim=[0 C(3)];
imTest21 = imagesc(H4_4,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
subplot(6,9,31)
clim=[0 C(4)];
imTest22 = imagesc(B2_4,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
subplot(6,9,32)
clim=[0 C(5)];
imTest23 = imagesc(B3_4,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
subplot(6,9,33)
clim=[0 C(6)];
imTest24 = imagesc(B4_4,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
subplot(6,9,34)
clim=[0 C(7)];
imTest25 = imagesc(F2_4,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
subplot(6,9,35)
clim=[0 C(8)];
imTest26 = imagesc(F3_4,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
subplot(6,9,36)
clim=[0 C(9)];
imTest27 = imagesc(F4_4,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
colormap jet;

ER = 32000;
D = [ER/206,ER/153,ER/156,ER/211,ER/211,ER/123,ER/210,ER/148,ER/149];
subplot(6,9,37)
clim=[0 D(1)];
imTest28 = imagesc(H2_5,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
subplot(6,9,38)
clim=[0 D(2)];
imTest29 = imagesc(H3_5,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
subplot(6,9,39)
clim=[0 D(3)];
imTest30 = imagesc(H4_5,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
subplot(6,9,40)
clim=[0 D(4)];
imTest31 = imagesc(B2_5,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
subplot(6,9,41)
clim=[0 D(5)];
imTest32 = imagesc(B3_5,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
subplot(6,9,42)
clim=[0 D(6)];
imTest33 = imagesc(B4_5,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
subplot(6,9,43)
clim=[0 D(7)];
imTest34 = imagesc(F2_5,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
subplot(6,9,44)
clim=[0 D(8)];
imTest35 = imagesc(F3_5,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
subplot(6,9,45)
clim=[0 D(9)];
imTest36 = imagesc(F4_5,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
colormap jet;


ER = 9000;
E = [ER/89,ER/60,ER/45,ER/55,ER/55,ER/55,ER/50,ER/50,ER/63];
subplot(6,9,46)
clim=[0 E(1)];
imTest37 = imagesc(H2_6,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
subplot(6,9,47)
clim=[0 E(2)];
imTest38 = imagesc(H3_6,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
subplot(6,9,48)
clim=[0 E(3)];
imTest39 = imagesc(H4_6,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
subplot(6,9,49)
clim=[0 E(4)];
imTest40 = imagesc(B2_6,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
subplot(6,9,50)
clim=[0 E(5)];
imTest41 = imagesc(B3_6,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
subplot(6,9,51)
clim=[0 E(6)];
imTest42 = imagesc(B4_6,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
subplot(6,9,52)
clim=[0 E(7)];
imTest43 = imagesc(F2_6,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
subplot(6,9,53)
clim=[0 E(8)];
imTest44 = imagesc(F3_6,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
subplot(6,9,54)
clim=[0 E(9)];
imTest45 = imagesc(F4_6,clim);
set(gca,'xtick',[],'ytick',[]) % x轴标签不显示
colormap jet;

for NN =1:54
    H(NN)=subplot(6,9,NN);%第NN张子图
    PPP=get(H(NN),'pos');%第NN张子图的当前位置
    PPP(1)=PPP(1)-0.01;%向右边延展0.04
    PPP(2)=PPP(2)-0.01;%向XIA方延展0.03
    PPP(3)=PPP(3)+0.01;%向右边延展0.04
    PPP(4)=PPP(4)+0.01;%向上方延展0.03
    set(H(NN),'pos',PPP);%根据新的边界设置。
    box off
    ax2 = axes('Position',get(gca,'Position'),...
           'XAxisLocation','top',...
           'YAxisLocation','right',...
           'Color','none',...
           'XColor','k','YColor','k');
    set(ax2,'YTick', []);
    set(ax2,'XTick', []);
end