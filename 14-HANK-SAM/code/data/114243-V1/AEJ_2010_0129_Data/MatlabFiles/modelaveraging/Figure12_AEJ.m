% HistoricalContributionMA.m

% This file plots historical contribution of MP shocks to business cycle
% fluctuations using 3 shocks measures:
%   a) R&R baseline shocks
%   b) R&R GARCH shocks
%   c) R&R TVC shocks
%   d) Smets-Wouters shocks

% This version uses estimates from MA estimation for inflation.

MPshockdata2;                       % loads data from 1965:1 1996:12, 
IP2=makelags(IP,60); CPI2=makelags(CPI,60); CPIc2=makelags(CPIc,60); PPI2=makelags(PPI,60); PiCPI2=makelags(PiCPI,60); PiCPIc2=makelags(PiCPIc,60); PiPPI2=makelags(PiPPI,60); FFR2=makelags(FFR,60); UE2=makelags(UE,60); PCOM2=makelags(PCOM,60); uRR2=makelags(uRR,60); 
p=CPI2; 
t=1970:1/12:1996+11/12;
warning('off')
figure('Name','MP-driven fluctuations from VAR and R&R')

%=========================================================================
% results for Baseline R&R Measure
%=========================================================================
subplot(3,4,1)
plot(t(13:length(t)),100*(IP2(13:length(IP2),1)-IP2(13:length(IP2),13)),'b:','Linewidth',2); hold on
                fit1=fitBreak(IP2(:,1)-IP2(:,2),[ones(length(uRR2),1) IP2(:,2:25)-IP2(:,3:26) uRR2(:,2:25)],60,26,0) ;
                fit3=100*(fit1(13:length(fit1))-fit1(1:length(fit1)-12));
plot(t(13:length(t)),fit3,'k','Linewidth',2); hold off
ylabel('Industrial Production')
title('Original R&R Shocks')
xlim([1970 1997])

subplot(3,4,5)
plot(t,UE2(:,1),'b:','Linewidth',2); hold on
                fit1=fitBreak(UE2(:,1),[ones(length(uRR2),1) UE2(:,2:25) uRR2(:,2:25)],60,26,1) ;
                
plot(t,fit1,'k','Linewidth',2); hold off
ylabel('Unemployment')
xlim([1970 1997])

subplot(3,4,9)
plot(t(13:length(t)),100*(p(13:length(IP2),1)-p(13:length(IP2),13)),'b:','Linewidth',2); hold on
fit2=[]; 
load(['MA3_AEJ' num2str(3) '_' num2str(1) '_' num2str(1)]);
   for j1=1:length(ARlags)
        j=ARlags(j1);
            for k1=1:length(MAlags)
                k=MAlags(k1);
                fit1=fitBreak(p(:,1)-p(:,2),[ones(length(uRR2),1) p(:,2:j+1)-p(:,3:j+2) uRR2(:,2:k+1)],60,j+2,0) ;
                fit3=100*(fit1(13:length(fit1))-fit1(1:length(fit1)-12));
                fit2=[fit2 fit3'];
            end
    end
fit1=fit2*pa1';
plot(t(13:length(t)),fit1,'k','Linewidth',2); hold off
ylabel('Annual Inflation')
xlim([1970 1997])

%=========================================================================
% results for R&R Garch
%=========================================================================
MPshockdataGARCH;
IP2=makelags(IP,maxlags); CPI2=makelags(CPI,maxlags); UE2=makelags(UE,maxlags); p2=CPI2; uRR2=makelags(uRR,maxlags);

subplot(3,4,2)
plot(t(13:length(t)),100*(IP2(13:length(IP2),1)-IP2(13:length(IP2),13)),'b:','Linewidth',2); hold on
                fit1=fitBreak(IP2(:,1)-IP2(:,2),[ones(length(uRR2),1) IP2(:,2:25)-IP2(:,3:26) uRR2(:,2:25)],60,26,0) ;
                fit3=100*(fit1(13:length(fit1))-fit1(1:length(fit1)-12));
plot(t(13:length(t)),fit3,'k','Linewidth',2); hold off
ylabel('Industrial Production')
title('GARCH R&R Shocks')
xlim([1970 1997])

subplot(3,4,6)
plot(t,UE2(:,1),'b:','Linewidth',2); hold on
fit2=[];
                fit1=fitBreak(UE2(:,1),[ones(length(uRR2),1) UE2(:,2:25) uRR2(:,2:25)],60,26,1) ;
                
plot(t,fit1,'k','Linewidth',2); hold off
ylabel('Unemployment')
xlim([1970 1997])

subplot(3,4,10)
plot(t(13:length(t)),100*(p(13:length(IP2),1)-p(13:length(IP2),13)),'b:','Linewidth',2); hold on
fit2=[]; 
load(['MA3_AEJ' num2str(3) '_' num2str(1) '_' num2str(2)]);
   for j1=1:length(ARlags)
        j=ARlags(j1);
            for k1=1:length(MAlags)
                k=MAlags(k1);
                fit1=fitBreak(p(:,1)-p(:,2),[ones(length(uRR2),1) p(:,2:j+1)-p(:,3:j+2) uRR2(:,2:k+1)],60,j+2,0) ;
                fit3=100*(fit1(13:length(fit1))-fit1(1:length(fit1)-12));
                fit2=[fit2 fit3'];
            end
    end
fit1=fit2*pa1';
plot(t(13:length(t)),fit1,'k','Linewidth',2); hold off
ylabel('Annual Inflation')
xlim([1970 1997])




%=========================================================================
% results for TVC approach
%=========================================================================
MPshockdataTVP;
IP2=makelags(IP,maxlags); CPI2=makelags(CPI,maxlags); UE2=makelags(UE,maxlags); p2=CPI2; uRR2=makelags(uRR,maxlags);

subplot(3,4,3)
plot(t(13:length(t)),100*(IP2(13:length(IP2),1)-IP2(13:length(IP2),13)),'b:','Linewidth',2); hold on
fit2=[]; 
                fit1=fitBreak(IP2(:,1)-IP2(:,2),[ones(length(uRR2),1) IP2(:,2:25)-IP2(:,3:26) uRR2(:,2:25)],60,26,0) ;
                fit3=100*(fit1(13:length(fit1))-fit1(1:length(fit1)-12));
plot(t(13:length(t)),fit3,'k','Linewidth',2); hold off
ylabel('Industrial Production')
title('TVC R&R Shocks')
xlim([1970 1997])

subplot(3,4,7)
plot(t,UE2(:,1),'b:','Linewidth',2); hold on
fit2=[];
                fit1=fitBreak(UE2(:,1),[ones(length(uRR2),1) UE2(:,2:25) uRR2(:,2:25)],60,26,1) ;
                
plot(t,fit1,'k','Linewidth',2); hold off
ylabel('Unemployment')
xlim([1970 1997])

subplot(3,4,11)
plot(t(13:length(t)),100*(p(13:length(IP2),1)-p(13:length(IP2),13)),'b:','Linewidth',2); hold on
fit2=[]; 
load(['MA3_AEJ' num2str(3) '_' num2str(1) '_' num2str(3)]);
   for j1=1:length(ARlags)
        j=ARlags(j1);
            for k1=1:length(MAlags)
                k=MAlags(k1);
                fit1=fitBreak(p(:,1)-p(:,2),[ones(length(uRR2),1) p(:,2:j+1)-p(:,3:j+2) uRR2(:,2:k+1)],60,j+2,0) ;
                fit3=100*(fit1(13:length(fit1))-fit1(1:length(fit1)-12));
               fit2=[fit2 fit3'];
            end
    end
fit1=fit2*pa1';
plot(t(13:length(t)),fit1,'k','Linewidth',2); hold off
ylabel('Annual Inflation')
xlim([1970 1997])


%==========================================================================
% results for Smets-Wouters
%==========================================================================
MPshockdataSW;                       % loads data from 1965:1 1996:12
IP2=makelags(IP,20); CPI2=makelags(CPI,20); UE2=makelags(UE,20);  uRR2=makelags(uSW,20); p2=CPI2;

t=1970:1/4:1996.75;
subplot(3,4,4)
fit2=[]; 
                fit1=fitBreakQ(IP2(:,1)-IP2(:,2),[ones(length(uRR2),1) IP2(:,2:9)-IP2(:,3:10) uRR2(:,2:9)],20,10,0) ;
                fit3=100*(fit1(5:length(fit1))-fit1(1:length(fit1)-4));

plot(t(5:length(t)),100*(IP2(5:length(IP2),1)-IP2(5:length(IP2),5)),'b:','Linewidth',2); hold on
plot(t(5:length(t)),fit3,'k','Linewidth',2); hold off
title('Smets-Wouters')
xlim([1970 1997])

subplot(3,4,8)
plot(t,UE2(:,1),'b:','Linewidth',2); hold on
                fit1=fitBreakQ(UE2(:,1),[ones(length(uRR2),1) UE2(:,2:9) uRR2(:,2:9)],20,10,1) ;
plot(t,fit1,'k','Linewidth',2); hold off
xlim([1970 1997])

subplot(3,4,12)
plot(t(5:length(t)),100*(p2(5:length(IP2),1)-p2(5:length(IP2),5)),'b:','Linewidth',2); hold on
fit2=[]; load(['MA3_AEJ' num2str(3) '_' num2str(1) '_' num2str(4)]);
   for j1=1:length(ARlags)
        j=ARlags(j1);
            for k1=1:length(MAlags)
                k=MAlags(k1);
                fit1=fitBreakQ(p2(:,1)-p2(:,2),[ones(length(uRR2),1) p2(:,2:j+1)-p2(:,3:j+2) uRR2(:,2:k+1)],20,j+2,0) ;
                fit3=100*(fit1(5:length(fit1))-fit1(1:length(fit1)-4));
                fit2=[fit2 fit3'];
            end
    end
fit1=fit2*pa1';
plot(t(5:length(t)),fit1,'k','Linewidth',2); hold off
xlim([1970 1997])



