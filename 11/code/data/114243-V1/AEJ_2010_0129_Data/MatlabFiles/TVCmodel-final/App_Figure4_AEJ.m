% This code computes the time-varying coefficients using the method
% developed in Boivin (JMCB 2006)
% Key modifications:
% a) allow for three breaks in the size of volatility:
%    pre-1979,
%    1979-1982,
%    post-1982;


% Coefficeints int he Taylor rule are assumed to follow random walk

clear
% close all
randn('seed',1234567)

num_sim_cov=1000; 
its=1000;         


%========================================================================
% Step 1: import data and construct RHS and LHS
%========================================================================
disp('Step 1: Import data and run OLS regressions ...')
import_data_greenbook;

Y=data_greenbook(:,4);
X=data_greenbook(:,5:10);
inflation=data_greenbook(:,11); % GDP deflator smoothed over current and past 12 quarters
T=length(Y);

trend_inflation=data_greenbook(:,12);
trend_inflation_date1=129;
trend_inflation_date2=157;

% create x-axis labels
labelfigs=[1 data_greenbook(1,2)];
stepfig=2;  % two year step for labels in figures
for i=2:T
    if labelfigs(end,2)+stepfig-1<data_greenbook(i,2)
        labelfigs=[labelfigs; i data_greenbook(i,2)];
    end
end

%========================================================================
% Step 2: estimate the size of the shocks to coefficients
%========================================================================
disp('Step 2: Run OLS regressions and estimate variance of shocks ...')
% The Stock and Watson (JASA 1998) parameter to the size of the shocks to
% coefficients in the Taylor rule. S&W method is used to have a median
% unbiased estimate for the size of the shock. Otherwise, the variances of
% innovations in coefs collapse to zero.

disp(['Quandt Liklihood ratio test statistic: ' num2str(QLR(Y,X))])
lambda=8; % This parameter is taken from Stock and Watson (JASA 1998)

% estimated Taylor rule:
% coef #1 = expected inflation rate, E(PI{t+1}|I_{t}), pi_tp
% coef #2 = current growth rate of output, Y{t}, gry_t
% coef #3 = current output gap, gap{t}, gap{t}, gap_t00
% coef #4 = FFR{t-1}
% coef #5 = FFR{t-2}
% coef #6 = constant terms

% ALLOW FOR THREE PERIODS WITH DIFFERENT VOLATILITIES:
% a) pre-1979
% b) 1979-1982
% c) post-1982

% a) pre-1979
X1=X(1:130,:);
Y1=Y(1:130,1);

beta=inv(X1'*X1)*(X1'*Y1); % OLS estimate of the policy reaction function across all periods
resid=Y1-X1*beta;

R=var(resid);   % variance of the error term

SigmaXX=(X1'*X1)/length(X1);

EXeeX=0; % White heteroskedasticity consistent estimate
for i=1:length(X1)
    EXeeX=EXeeX+resid(i)^2*X1(i,:)'*X1(i,:);
end
EXeeX=EXeeX/length(X1);

SigmaVV1=inv(SigmaXX)*EXeeX*inv(SigmaXX);

% estimate of innovations to coefficients
SigmaBB1=(lambda/T)^2*SigmaVV1;
R1=R;

%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% a) 1979-1982
X1=X(131:159,:);
Y1=Y(131:159,1);

beta=inv(X1'*X1)*(X1'*Y1); % OLS estimate of the policy reaction function across all periods
resid=(Y1-X1*beta);

R=var(resid);   % variance of the error term

SigmaXX=(X1'*X1)/length(X1);

EXeeX=0; % White heteroskedasticity consistent estimate
for i=1:length(X1)
    EXeeX=EXeeX+resid(i)^2*X1(i,:)'*X1(i,:);
end
EXeeX=EXeeX/length(X1);

SigmaVV2=inv(SigmaXX)*EXeeX*inv(SigmaXX);

% estimate of innovations to coefficients
SigmaBB2=(lambda/T)^2*SigmaVV2;
R2=R;

%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% a) post-1982
X1=X(160:end,:);
Y1=Y(160:end,1);

beta=inv(X1'*X1)*(X1'*Y1); % OLS estimate of the policy reaction function across all periods
resid=Y1-X1*beta;

R=var(resid);   % variance of the error term

SigmaXX=(X1'*X1)/length(X1);

EXeeX=0; % White heteroskedasticity consistent estimate
for i=1:length(X1)
    EXeeX=EXeeX+resid(i)^2*X1(i,:)'*X1(i,:);
end
EXeeX=EXeeX/length(X1);

SigmaVV3=inv(SigmaXX)*EXeeX*inv(SigmaXX);

% estimate of innovations to coefficients
SigmaBB3=(lambda/T)^2*SigmaVV3;
R3=R;


R2=R1;
R3=R1;
SigmaBB2=SigmaBB1;
SigmaBB3=SigmaBB1;





%========================================================================
% Step 4: run the Kalman filter and the smoother
%========================================================================
disp('Step 4: Run Kalman filter and smoother ...')
[SEstate,SVstate,KFEstate,KFVstate,KFresid,MLE]=KalmanFSbreaks(Y,X,beta,SigmaBB1,R1,SigmaBB2,R2,SigmaBB3,R3);
% compute standard errors
for i=1:T
    SVstate_se(i,:)=diag(sqrt(squeeze(SVstate(i,:,:))))';
end

for i=1:T
    KFVstate_se(i,:)=diag(sqrt(squeeze(KFVstate(i,:,:))))';
end

figure(1)
for j=1:6
    subplot(3,2,j)
    plot(SEstate(:,j),'k','Linewidth',2)
    hold on
    plot(SEstate(:,j)+SVstate_se(:,j),'b:')
    plot(SEstate(:,j)-SVstate_se(:,j),'b:')
    hold off
    set(gca,'XTick',labelfigs(:,1))
    set(gca,'XTickLabel',labelfigs(:,2))
    xlim([1 T])
    if j==1
        title('Inflation Response')
    elseif j==2
        title('Output Growth Response')
    elseif j==3
        title('Unemployment')
    elseif j==4
        title('Interest Smoothing AR(1)')
    elseif j==5
        title('Interest Smoothing AR(2)')
    else
        title('Intercept')
    end

end

figure(2)
lrpi=SEstate(:,1)./(-SEstate(:,4)-SEstate(:,5)); % long-run inflation response
plot(lrpi)
set(gca,'XTick',labelfigs(:,1))
set(gca,'XTickLabel',labelfigs(:,2))
xlim([1 T])

