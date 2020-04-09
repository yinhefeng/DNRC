clear
clc
close all

% data loading (here we use the AR dataset as an example)
load('AR_DAT.mat');

% parameter setting
par.nClass = length(unique(trainlabels)); % the number of classes in the subset of AR database
dim = [54 120 300];
lambda = [1e-1,1e-1,1];
mu= 1e-1;

% data and labels for training and test samples
%--------------------------------------------------------------------------
Tr_DAT   =   double(NewTrain_DAT(:,trainlabels<=par.nClass));
trls     =   trainlabels(trainlabels<=par.nClass);
Tt_DAT   =   double(NewTest_DAT(:,testlabels<=par.nClass));
ttls     =   testlabels(testlabels<=par.nClass);
clear NewTest_DAT NewTrain_DAT testlabels trainlabels


test_tol = size(Tt_DAT,2);
reg_rate = zeros(1,length(dim));
kk = 1;

param = [];
param.mu = mu;

for eigen_num=dim
    % eigenface extracting
    [disc_set,disc_value,Mean_Image]  =  Eigenface_f(Tr_DAT,eigen_num);
    tr_dat  =  disc_set'*Tr_DAT;
    tt_dat  =  disc_set'*Tt_DAT;
    
    % unit L2 norm
    tr_dat = normc(tr_dat);
    tt_dat = normc(tt_dat);
    % tr_dat  =  tr_dat./( repmat(sqrt(sum(tr_dat.*tr_dat)), [par.nDim,1]) );
    % tt_dat  =  tt_dat./( repmat(sqrt(sum(tt_dat.*tt_dat)), [par.nDim,1]) );
    
    ID = zeros(1,test_tol);
    X = tr_dat;
    Y = tt_dat;
    param.lambda = lambda(kk);
    % coding
    [~,C] = DNRC(X, Y, param);
    for i=1:test_tol
        y = tt_dat(:,i);
        c = C(:,i);
        residual = DNRC_res(X,y,c,trls);
        
        % classification
        [~,index]=min(residual);
        ID(i)=index;
    end
    
    cornum      =   sum(ID==ttls);
    reg_rate(kk)         =   cornum/length(ttls); % recognition rate
    kk = kk+1;
end

% display recognition result
disp([dim;reg_rate])