% AICsim.m

% this function takes regression equation, simulates data from it, selects 
% optimal lag length via AIC for each generated sample, and calculates the
% fraction of draws which match pre-specified lag specification.

function [p1, P]=AICsimGEN1(Y,X,N,lagAIC,ARlags,levelindex,periods,ARvec,MAvec,minindex);

% details
% Y is RHS variables
% X is matrix of RHS variables.  Includes: constant, ARlags of LHS, and
%                                          MAlags of shock variables
% N is # of simulations to do
% lagAIC is lag specification we want to match
% ARlags is number of lags of LHS variable included in X

% ARvec is set of AR lags considered
% MAvec is set of lags of shock considered
% minindex determines whether we search for min of IRF (=1) or max

% p is frequency of observing lagAIC for each lag specification
% P is estimated peak effect for each lag specification for different true
% lag specifications.

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
    
    a1max=-1000; lagAIC1=[1 1]; ind=1;
    for j1=1:length(ARvec)
        j=ARvec(j1);
        for k1=1:length(MAvec)
            k=MAvec(k1);
            % AIC selection
            a1=aic(Y1(:,1),[ones(length(Y1),1) Y1(:,2:1+j) mp1(:,2:1+k)]);
            if a1>a1max   a1max=a1; lagAIC1=[j k]; end
            
            % estimate peak effect from each lag specification
            irfA=impulse(Y1(:,1),[ones(length(Y1),1) Y1(:,2:1+j) mp1(:,2:1+k)],periods,j+2,levelindex);
            if minindex==1
                P1(it,ind)=min(irfA); ind=ind+1;
            else
                P1(it,ind)=max(irfA); ind=ind+1;
            end
        end
    end
    %display(['Sim: ' num2str(it) ' AIC Selected: ' num2str(lagAIC1)])              %TEMP
    if lagAIC1==lagAIC
        indic(it,1)=1;
    else
        indic(it,1)=0;
    end
end

p1=sum(indic(:,1))/N;

P=mean(P1);

return
    
