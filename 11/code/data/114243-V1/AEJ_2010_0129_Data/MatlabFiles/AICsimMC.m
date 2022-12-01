% AICsim.m

% this function takes regression equation, simulates data from it, selects 
% optimal lag length via AIC for each generated sample, and calculates the
% fraction of draws which match pre-specified lag specification.

function [p1, p2, p3, sd]=AICsimMC(Y,X,N,lagAIC,lagBIC,ARlags,levelindex,periods,jmin,jmax,kmin,kmax);

% details
% Y is RHS variables
% X is matrix of RHS variables.  Includes: constant, ARlags of LHS, and
%                                          MAlags of shock variables
% N is # of simulations to do
% lagAIC is lag specification we want to match
% ARlags is number of lags of LHS variable included in X
I=100;


% parameters for simulation
Beta=inv(X'*X)*(X'*Y);                                              % coefficients
ARcoefs=Beta(2:ARlags+1); MAcoefs=Beta(1+ARlags+1:length(Beta));    % AR and MA coefficients
res1=Y-X*Beta;                                                       % residuals
MAlags=cols(X)-1-ARlags;
mpshocksDAT=[0; X(1:length(X)-1,1+ARlags+1)];         % original shocks (makes use of fact that only lagged shocks are used in regression
maxlags=max(ARlags,MAlags);
T=length(X);

for it=1:N
    clear mpshocks Y Xa mp1 Y1
    
    % make data
    for j=1:I+T
        e=ceil(length(res1)*rand(1));
        res(j)=res1(e);                     % this generates residuals
        mpshocks(j,1)=mpshocksDAT(e,1);     % this generates MP shocks
    end
    
    mpshocks=[zeros(maxlags,1);mpshocks]; mpshocks=makelags(mpshocks,maxlags);
    Xa=X(1,2:1+ARlags);
    
    for j=1:I+T
        Y(j,1)=Beta(1)+Xa(j,1:ARlags)*ARcoefs+mpshocks(j,2:MAlags+1)*MAcoefs+res(j);
        Xa(j+1,1:ARlags)=[Y(j,1) Xa(j,1:ARlags-1)];
    end
    
    % format data for estimation
    Y1=Y(I+1-45:I+T,1); Y1=makelags(Y1,45);
    mp1=mpshocks(I+1-45:I+T,1);   mp1=makelags(mp1,45);
    
    % estimate impulse response given correct specification
    irf(it,:)=impulse(Y1(:,1),[ones(length(Y1),1) Y1(:,2:1+ARlags) mp1(:,2:1+MAlags)],periods,ARlags+2,levelindex);
    
    a1max=0; lagAIC1=[1 1]; b1max=0; lagBIC1=[1 1];
    for j=jmin:jmax
        for k=kmin:kmax
            % AIC selection
            a1=aic(Y1(:,1),[ones(length(Y1),1) Y1(:,2:1+j) mp1(:,2:1+k)]);
            if a1>a1max   a1max=a1; lagAIC1=[j k]; end
            % BIC selection
            b1=bic(Y1(:,1),[ones(length(Y1),1) Y1(:,2:1+j) mp1(:,2:1+k)]);
            if b1>b1max   b1max=b1; lagBIC1=[j k]; end
        end
    end
    %display(['Sim: ' num2str(it) ' AIC Selected: ' num2str(lagAIC1)])              %TEMP
    if lagAIC1==lagAIC
        indic(it,1)=1;
    else
        indic(it,1)=0;
    end
    if lagBIC1==lagBIC
        indic(it,2)=1;
    else
        indic(it,2)=0;
    end
    if lagAIC1==lagAIC & lagBIC1==lagBIC
        indic(it,3)=1;
    else
        indic(it,3)=0;
    end
end

p1=sum(indic(:,1))/N;
p2=sum(indic(:,2))/N;
p3=sum(indic(:,3))/N;
sd=std(irf)';
return
    
