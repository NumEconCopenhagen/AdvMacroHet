%%%   FigureMC.m

%%%   This file does MC of IP equation and shows AIB/BIC underfitting 

clear all
%%%   This version draws from actual MP shocks and residuals with
%%%   replacement.
seedn=23;    %TEMP

its=1000;
I=50;           % burn in period for simulations
T=320;          % length of sample in simulations
periods=36;     % length of IRF's for MSE calculatios
coefs=12;       % set to 12 or 24
levelindex=0;
minindex=1;
%N=200;           % number of simulations for bootstrap procedure.
jmin=24;
jmax=24;
kmin=3;
kmax=36;
incr=3;

rand('seed',1+seedn);
randn('seed',1+seedn);

%==========================================================================
% Part 1: Monte Carlo of AIC/BIC for different true lag lengths of MA
%==========================================================================

MPshockdata2;                       % loads data from 1965:1 1996:12
IP2=makelags(IP,60); CPI2=makelags(CPI,60); UE2=makelags(UE,60); p=CPI2; uRR2=makelags(uRR,60); 
Y=IP2(:,1)-IP2(:,2);
X=[ones(length(uRR2),1) IP2(:,2:coefs+1)-IP2(:,3:coefs+2) uRR2(:,2:coefs+1)];
Beta=inv(X'*X)*(X'*Y);
ARcoefs=Beta(2:coefs+1); MAcoefs=Beta(1+coefs+1:length(Beta));
resdip=Y-X*Beta;

% calculate true IRF from specified coefs
beta=[0;ARcoefs;MAcoefs]; Xa=zeros(1,length(beta)-1); z(1)=0; 
if coefs==12 shockpos=14; else shockpos=26; end
for j=2:periods
    Xa(j,1)=z(j-1,1);
    for i=2:length(beta)-1
        Xa(j,i)=Xa(j-1,i-1);
    end
    if j==2
        Xa(j,shockpos-1)=1;
    else
        Xa(j,shockpos-1)=0;
    end
    z(j,1)=Xa(j,:)*beta(2:length(beta));
end
    irf(1)=z(1);
    for j=2:length(z)
        irf(j)=irf(j-1)+z(j);
    end
    % irf is true IRF
ind=1;    
for kind=kmin:incr:kmax
    %coefs=kind; jmin=kind; jmax=kind;
    X=[ones(length(uRR2),1) IP2(:,2:coefs+1)-IP2(:,3:coefs+2) uRR2(:,2:kind+1)];
    Beta=inv(X'*X)*(X'*Y);
    ARcoefs=Beta(2:coefs+1); MAcoefs=Beta(1+coefs+1:length(Beta));
    resdip=Y-X*Beta;
for it=1:its
    
    clear mpshocks dip
    
    
    % make data, drawing shocks from residuals and actual policy shocks.
    for j=1:I+T
        e=ceil(length(resdip)*rand(1));
        res(j)=resdip(e);
        mpshocks(j,1)=uRR2(e,1);
    end
    
    mpshocks=[zeros(max(kind,coefs),1);mpshocks]; mpshocks=makelags(mpshocks,max(kind,coefs));
    dip=[IP2(1,2:max(kind,coefs)+1)-IP2(1,3:max(kind,coefs)+2) 0]';
    dip=makelags(dip,max(kind,coefs));
    for j=1:I+T
        dip(j,1)=Beta(1)+dip(j,2:coefs+1)*ARcoefs+mpshocks(j,2:kind+1)*MAcoefs+res(j);
        dip(j+1,2:coefs+1)=dip(j,1:coefs);
    end
    
    
    % format data for estimation
    dip1=dip(I+1-45:I+T,1); dip1=makelags(dip1,45);
    mp1=mpshocks(I+1-45:I+T,1);   mp1=makelags(mp1,45);
    
    % AIC/BIC selection for simulated data
    a1max=0; b1max=0; lagAIC=[1 1]; lagBIC=[1 1]; 
    for j=jmin:incr:jmax
        for k=kmin:incr:kmax
            b1=bic(dip1(:,1),[ones(length(dip1),1) dip1(:,2:1+j) mp1(:,2:1+k)]);
            a1=aic(dip1(:,1),[ones(length(dip1),1) dip1(:,2:1+j) mp1(:,2:1+k)]);
            if b1>b1max   b1max=b1; lagBIC=[j k]; end
            if a1>a1max   a1max=a1; lagAIC=[j k]; end
        end
    end
    AIC1(it,:)=lagAIC;
    BIC1(it,:)=lagBIC;
        
end

AIC(1,ind)=mean(AIC1(:,2));
BIC(1,ind)=mean(BIC1(:,2));
AICstd(1,ind)=std(AIC1(:,2));
BICstd(1,ind)=std(BIC1(:,2));

AIC1=sort(AIC1);
BIC1=sort(BIC1);
AIC975(1,ind)=AIC1(floor(.975*it),2);
AIC025(1,ind)=AIC1(floor(.025*it),2);
AIC840(1,ind)=AIC1(floor(.84*it),2);
AIC160(1,ind)=AIC1(floor(.16*it),2);
BIC975(1,ind)=BIC1(floor(.975*it),2);
BIC025(1,ind)=BIC1(floor(.025*it),2);

ind=ind+1;
%clear AIC1 BIC1
end

k1=kmin:incr:kmax;
figure(1)
%subplot(2,1,1)
plot(k1,AIC,'k--','Linewidth',2)
hold on
plot(k1,AIC975,'k:','Linewidth',2)
plot(k1,AIC025,'k:','Linewidth',2)
plot(k1,BIC,'b-.','Linewidth',2)
plot(k1,k1,'k')
hold off
xlabel('True Lag Length for Shocks')
ylabel('Selected Lag Length from AIC/BIC')

