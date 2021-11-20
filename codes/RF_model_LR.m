%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Model: Random Forest
% Watershed: Logan River watershed
% Lookback: 365 (hand crafted input features within 365 days)
% Reference: Xu, T., Longyang, Q., Tyson, C., Zeng, R., and Neilson, B.T., Hybrid physically- based and deep learning modeling
% of a snow dominated, mountainous, karst watershed, under review.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; clc; close all
%% data preparation - dividing into training & test
opt.ntr = datenum('10/01/2007') - datenum('10/01/1981'); % training period

% "data" conent: column 1-11: input features described in Table S2,
% Supporting Information, Xu et al., under revie; column 15: streamflow
% target
load xy_dat_LR_lump
opt.y_ind = 12;
opt.x_ind = 1:11;

% predicting Q_t+1
tmp = dat(:,opt.y_ind);
dat(end,:) = [];
dat(:,opt.y_ind) = tmp(2:end);
t(1) = [];  

ind_te = datenum('10/01/2007') - datenum('10/01/1981')+ 1 : datenum('10/01/2018') - datenum('10/02/1981');
ind_tr = 1:opt.ntr;

x_tr = dat(ind_tr,opt.x_ind);
y_tr = dat(ind_tr,opt.y_ind);
x_te = dat(ind_te,opt.x_ind);
y_te = dat(ind_te,opt.y_ind);

%% Random forest model
ntree = 500; % not sensitive as long as big enough
B = TreeBagger(ntree,x_tr,y_tr,'method','regression','OOBVarImp','on','oobpred','on',...
    'NumPredictorsToSample','all','MinLeafSize',10);  % make sure to consider all variables at each split calculate a more robust predictor importance
err_oob = oobError(B,'mode','ensemble').^0.5;
% extract oob variable importance measure
var_imp = B.OOBPermutedVarDeltaError./sum(B.OOBPermutedVarDeltaError);
var_imp = var_imp(:);
% train again
% leaf size is a tunable parameter
B = TreeBagger(ntree,x_tr,y_tr,'method','regression','MinLeafSize',50);
% Test
ypred_tr = predict(B,x_tr);
ypred = predict(B,x_te);

var_name = {'Dryspell','$R_t$ [mm]','$R_{t-1}$ [mm]','$\bar{R}_{t-15}$ [mm]','$\bar{R}_{t-28}$ [mm]',...
    '$\bar{R}_{t-40}$ [mm]','$\sum_{i=1}^{28}{R_{t-i}}$ [mm]','$\sum_{i=1}^{56}{R_{t-i}}$ [mm]',...
    '$\sum_{i=1}^{150}{R_{t-i}}$ [mm]','$\sum_{i=0}^{3}{PET_{t-i}}$ [mm]','$\sum_{i=0}^{151}{PET_{t-i}}$ [mm]'};

%% calculate partial dependence of each variable to complement variable importance score
figure;
for i = 1:length(opt.x_ind)
    subplot(3,4,i);
    box on
    plotPartialDependence(B,i);
    title('');
    xlabel(var_name{i},'Interpreter','latex');
    ylabel('Q');
end
