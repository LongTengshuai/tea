%+++ Guide of iRF method for many runs+++%
runs=20; %the number of iRF runs
K=10; % the group number for cross validation.
A=10;% the maximal principle component
method='center';%pretreatment method
ratio=0.8; % training set:80%, test set:20%
[mx,nx]=size(X);
mtrain=ceil(mx*ratio);
mtest=mx-mtrain;

% KS spilt dataset into training set and test set
[Xtrain,Xtest,Ytrain,Ytest]=ks1(XX,Y,ceil(mx*ratio));
XXtrain=Xtrain;
XXtest=Xtest;
for i=1:runs
    F=iRF(XXtrain,Ytrain,10000,20,50,10,'center');
    probability{i}=F.probability;
end
%+++ XXtrain: training set
%+++ Ytrain: The reponse vector of the training set
%+++ 10000: the number of iterations
%+++ 20: the fixed window size to move over the whole spectra
%+++ 50: the initialized number of sub-intervasl.
%+++ 10: the maximal principle component.
%+++ 'center': pretreatment method.


% compute the RMSECV of the union of the ranked intervals from 10th to the last one 
sumprobability=zeros(1,size(F.intervals,2));
for iii=1:runs
    sumprobability=probability{iii}+sumprobability;
end

averageprobability=sumprobability./iii;
[sorted,Intervalsrank]=sort(-averageprobability);
k=1;
for j=10: size(F.intervals,2)    
    Utemp=F.intervals{Intervalsrank(1)};      
    for iii=2:j        
        Utemp=union(Utemp,F.intervals{Intervalsrank(iii)});      
    end     
    vsel_temp{1,k}=Utemp;   
    Xtrain=XXtrain(:,Utemp); 
    CV=plscvfold(Xtrain,Ytrain,A,K,'center',0);
    RMSECV(k)=CV.RMSECV;
    fprintf('The %dth of %d circle finished.\n',k,size(F.intervals,2)-10+1) 
    k=k+1;
end
% choose the intervals with the lowest RMSECV
[num,index]=min(RMSECV);
% the union of selected intervals 
vsel=vsel_temp{index};
