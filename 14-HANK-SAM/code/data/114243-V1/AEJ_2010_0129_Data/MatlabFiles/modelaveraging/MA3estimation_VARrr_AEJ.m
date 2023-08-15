% MAestimation_testVAR.m

% this file tests the model-averaging procedure for IRF's using AIC for VARs.
clear all
% parameters
N=1000;              % number of simulations in each MC
periods=60;           % length of IRF's
Lags=[3 6 9 12 15 18 21 24 27 30 33 36];  % lag lengths considered

%========================================================================
%%%     STEP 1:  data formatting and initial AIC selection
%========================================================================
MPshockdata2;                       % loads data from 1965:1 1996:12
RRcumul;                            % loads R&R cumulative shock (shRR), 1965:1 1996:12
IP2=makelags(IP,60); CPI2=makelags(CPI,60); CPIc2=makelags(CPIc,60); PPI2=makelags(PPI,60); PiCPI2=makelags(PiCPI,60); PiCPIc2=makelags(PiCPIc,60); PiPPI2=makelags(PiPPI,60); FFR2=makelags(FFR,60); UE2=makelags(UE,60); PCOM2=makelags(PCOM,60); p=CPI2; uRR2=makelags(uRR,60); shRR2=makelags(shRR,60); p2=CPI2;
options.irfhor=periods; options.vdechor=periods; options.constant=1;
FFR2=shRR2;

for z=1:2
    if z==2
        IP2=[IP2(1:(1979+8/12-1969-11/12)*12,:) ; IP2((1984-1969-11/12)*12:length(IP2),:)];
        UE2=[UE2(1:(1979+8/12-1969-11/12)*12,:) ; UE2((1984-1969-11/12)*12:length(UE2),:)];
        p2= [  p2(1:(1979+8/12-1969-11/12)*12,:) ;   p2((1984-1969-11/12)*12:length(p2),:)];
        PCOM2=[PCOM2(1:(1979+8/12-1969-11/12)*12,:) ; PCOM2((1984-1969-11/12)*12:length(PCOM2),:)];
        FFR2=[FFR2(1:(1979+8/12-1969-11/12)*12,:) ; FFR2((1984-1969-11/12)*12:length(FFR2),:)];
    end


a1max=0; b1max=0;
for j1=1:length(Lags)
    j=Lags(j1);
            % AIC selection
            dat=[IP2(:,1:j+1) UE2(:,1:j+1) p2(:,1:j+1) PCOM2(:,1:j+1) FFR2(:,1:j+1) ]; 
            VARa=aicVAR(dat,j,options);
            VARb=bicVAR(dat,j,options);
            if VARa>a1max   a1max=VARa;  lagAIC=j; end
            if VARb>b1max   b1max=VARb;  lagBIC=j; end
end
dataic=[IP2(:,1:lagAIC+1) UE2(:,1:lagAIC+1) p2(:,1:lagAIC+1) PCOM2(:,1:lagAIC+1) FFR2(:,1:lagAIC+1) ]; 
outaic=varcg(dataic,lagAIC,options);                                                
irfaic=outaic.irf; % irf of AIC specification

datbic=[IP2(:,1:lagBIC+1) UE2(:,1:lagBIC+1) p2(:,1:lagBIC+1) PCOM2(:,1:lagBIC+1) FFR2(:,1:lagBIC+1) ]; 
outbic=varcg(datbic,lagBIC,options);                                                
irfbic=outbic.irf; % irf of AIC specification

% step 1 complete: lagAIC is optimal lag specification.


%========================================================================
%%%     STEP 2:  iterate through lag specifications and calculate fraction
%%%     of draws yielding same AIC as in data.  Also, compute mean
%%%     estimated effects for each lag length under null of each lag
%%%     length.
%========================================================================
ind=1; p=[]; P1=[]; P2=[]; P3=[];
for j1=1:length(Lags)
    j=Lags(j1);
        dat1=[IP2(:,1:j+1) UE2(:,1:j+1) p2(:,1:j+1) PCOM2(:,1:j+1) FFR2(:,1:j+1) ]; 
        [p1 P11 P22 P33]=AICsimVAR(dat1,N,lagAIC,j,Lags);
        p=[p; p1]; P1=[P1; P11]; P2=[P2; P22]; P3=[P3; P33];
        out1=varcg(dat1,j,options);
        irfip(:,ind)=out1.irf(:,1,5);
        irfue(:,ind)=out1.irf(:,2,5);
        irfp(:,ind)=out1.irf(:,3,5);

        ind=ind+1
end
% step 2 complete

% conditional probabilities
p=p/sum(p);


if z==1
    irfaic1=irfaic;     % these are impulse responses under AIC selection
    irfbic1=irfbic;     % these are impulse responses under BIC selection
    irfip1=irfip;       % this is matrix of IP IRF's for all lags
    irfue1=irfue;       % this is matrix of UE IRF's for all lags
    irfp1=irfp;         % this is matrix of P IRF's for all lags
    pa=p;               % these are bootstrapped probabilities of matching AIC lag
    lagAIC1=lagAIC; lagBIC1=lagBIC;
end

end
save('MA3estimation_VARrr_AEJ')





%=======================================================================
%   STEP 3: Plot IRF's of model-averaging
%=======================================================================

figure(1)
subplot(2,3,1)
plot(irf1(:,1,5),'k','linewidth',2)
hold on
plot((irfip1*pa),'b:','linewidth',2)
plot((irfip1*w1(:,1)),'r--','linewidth',2)
hold off

subplot(2,3,2)
plot(irf1(:,2,5),'k','linewidth',2)
hold on
plot((irfue1*pa),'b:','linewidth',2)
plot((irfue1*w1(:,2)),'r--','linewidth',2)
hold off

subplot(2,3,3)
plot(irf1(:,3,5),'k','linewidth',2)
hold on
plot((irfp1*pa),'b:','linewidth',2)
plot((irfp1*w1(:,3)),'r--','linewidth',2)
hold off

subplot(2,3,4)
plot(irf(:,1,5),'k','linewidth',2)
hold on
plot((irfip*p),'b:','linewidth',2)
plot((irfip*w(:,1)),'r--','linewidth',2)
hold off

subplot(2,3,5)
plot(irf(:,2,5),'k','linewidth',2)
hold on
plot((irfue*p),'b:','linewidth',2)
plot((irfue*w(:,2)),'r--','linewidth',2)
hold off

subplot(2,3,6)
plot(irf(:,3,5),'k','linewidth',2)
hold on
plot((irfp*p),'b:','linewidth',2)
plot((irfp*w(:,3)),'r--','linewidth',2)
hold off




