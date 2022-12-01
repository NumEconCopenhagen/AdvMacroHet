% estimationOutliers.m

% this file estimates maximum effects of MP shocks on IP, UE, P, and FFR
% using 3 measures of MP shocks and dropping rolling 3-mo periods.
% all estimation done using univariate approach.
clear all

%%% Estimation options
pmeasureVAR=1;                      % select price measure to include in baseline VAR to generate VAR MP shocks: 1 CPI, 2 CPI less housing, 3 PPI, 4 annual CPI inflation, 5 annual CPIc inflation, 6 annual PPI inflation
pmeasureIRF=1;                      % select price measure to include in IRF's: 1 CPI, 2 CPI less housing, 3 PPI, 4 annual CPI inflation, 5 annual CPIc inflation, 6 annual PPI inflation

MPshockdata2;                        % load macro data: IP, UE, CPI, CPIc, PPI, FFR, PCOM: 1968:1-1996:12, monthly
RRcumul;                            % loads R&R cumulative shock (shRR), 1965:1 1996:12
IP2=makelags(IP,60); CPI2=makelags(CPI,60); CPIc2=makelags(CPIc,60); PPI2=makelags(PPI,60); PiCPI2=makelags(PiCPI,60); PiCPIc2=makelags(PiCPIc,60); PiPPI2=makelags(PiPPI,60); FFR2=makelags(FFR,60); UE2=makelags(UE,60); PCOM2=makelags(PCOM,60); shRR2=makelags(shRR,60); uRR2=makelags(uRR,60); 
if pmeasureVAR==1 p=CPI2; elseif pmeasureVAR==2 p=CPIc2; elseif pmeasureVAR==3 p=PPI2; elseif pmeasureVAR==4 p=PiCPI2; elseif pmeasureVAR==5 p=PiCPIc2; else p=PiPPI2; end  % select which measure of prices to include in baseline VAR

dat2=[IP2(:,1:13) UE2(:,1:13) p(:,1:13) PCOM2(:,1:13) FFR2(:,1:13) ];        
dat3=[IP2(:,1:13) UE2(:,1:13) p(:,1:13) PCOM2(:,1:13) shRR2(:,1:13)  ];        
options.irfhor=60; options.vdechor=60; options.constant=1;

out2=varcg(dat2,12,options);        % basic VAR uses 12 lags (1 year): estimation of VAR and univariate processes start in 1970:1                                             
out3=varcg(dat3,12,options);        % R&R VAR uses 12 lags (1 year): estimation of VAR and univariate processes start in 1970:1                                             
uVAR=out2.u(:,5);
uVARrr=out3.u(:,5);
uVAR=[zeros(60,1);uVAR];     
uVARrr=[zeros(60,1);uVARrr]; 

% 3 shock series: uVAR1: baseline VAR shocks, uVAR2: R&R VAR shocks, uRR2: baseline R&R shocks
t=1970:1/12:1996+11/12;
    for j=1:length(uRR2)
        % generate shocks set to zero over rolling 3-mo intervals
        uRR1=uRR;                uVAR1=uVAR;                uVAR2=uVARrr;
        uRR1(60+j-2:60+j)=0;     uVAR1(60+j-2:60+j)=0;      uVAR2(60+j-2:60+j)=0;
        uRR1=makelags(uRR1,60);  uVAR1=makelags(uVAR1,60);  uVAR2=makelags(uVAR2,60);
        
        % estimate IRF's using common univariate approach and extract maximal responses
        fitIP=impulse(IP2(:,1)-IP2(:,2),[ones(length(uRR2),1) IP2(:,2:25)-IP2(:,3:26) uRR1(:,2:37)],60,26,0) ;
        maxIPrr(j)=min(fitIP);
        fitIP=impulse(IP2(:,1)-IP2(:,2),[ones(length(uRR2),1) IP2(:,2:25)-IP2(:,3:26) uVAR1(:,2:37)],60,26,0) ;
        maxIP1(j)=min(fitIP);
        fitIP=impulse(IP2(:,1)-IP2(:,2),[ones(length(uRR2),1) IP2(:,2:25)-IP2(:,3:26) uVAR2(:,2:37)],60,26,0) ;
        maxIP2(j)=min(fitIP);
        
        fitUE=impulse(UE2(:,1),[ones(length(uRR2),1) UE2(:,2:25) uRR1(:,2:37)],60,26,1) ;
        maxUErr(j)=max(fitUE);
        fitUE=impulse(UE2(:,1),[ones(length(uRR2),1) UE2(:,2:25) uVAR1(:,2:37)],60,26,1) ;
        maxUE1(j)=max(fitUE);
        fitUE=impulse(UE2(:,1),[ones(length(uRR2),1) UE2(:,2:25) uVAR2(:,2:37)],60,26,1) ;
        maxUE2(j)=max(fitUE);
        
        fitP=impulse(p(:,1)-p(:,2),[ones(length(uRR2),1) p(:,2:25)-p(:,3:26) uRR1(:,2:49)],60,26,0) ;
        maxPrr(j)=min(fitP);
        fitP=impulse(p(:,1)-p(:,2),[ones(length(uRR2),1) p(:,2:25)-p(:,3:26) uVAR1(:,2:49)],60,26,0) ;
        maxP1(j)=min(fitP);
        fitP=impulse(p(:,1)-p(:,2),[ones(length(uRR2),1) p(:,2:25)-p(:,3:26) uVAR2(:,2:49)],60,26,0) ;
        maxP2(j)=min(fitP);
        
        fitFFR=impulseA(FFR2(:,1),[ones(length(uRR2),1) FFR2(:,2:25) uRR1(:,1:37)],60,26,1) ;        
        maxFFRrr(j)=sum(fitFFR(1:12));
        fitFFR=impulseA(FFR2(:,1),[ones(length(uRR2),1) FFR2(:,2:25) uVAR1(:,1:37)],60,26,1) ;        
        maxFFR1(j)=sum(fitFFR(1:12));
        fitFFR=impulseA(FFR2(:,1),[ones(length(uRR2),1) FFR2(:,2:25) uVAR2(:,1:37)],60,26,1) ;        
        maxFFR2(j)=sum(fitFFR(1:12));
        
    end
    
    figure('Name','Outlier-Sensitivity')
    subplot(2,2,1)
    plot(t,maxIPrr,'k','Linewidth',2)
    hold on
    plot(t,maxIP1,'b:','Linewidth',1)
    plot(t,maxIP2,'k','Linewidth',1)
    hold off
    title('Industrial Production')
    ylabel('Peak Drop in Industrial Production')

    subplot(2,2,2)
    plot(t,maxUErr,'k','Linewidth',2)
    hold on
    plot(t,maxUE1,'b:','Linewidth',1)
    plot(t,maxUE2,'k','Linewidth',1)
    hold off
    title('Unemployment')
    ylabel('Peak Rise in Unemployment')

    subplot(2,2,3)
    plot(t,maxPrr,'k','Linewidth',2)
    hold on
    plot(t,maxP1,'b:','Linewidth',1)
    plot(t,maxP2,'k','Linewidth',1)
    hold off
    title('Prices')
    ylabel('Peak Drop in Prices')
    
    subplot(2,2,4)
    plot(t,maxFFRrr,'k','Linewidth',2)
    hold on
    plot(t,maxFFR1,'b:','Linewidth',1)
    plot(t,maxFFR2,'k','Linewidth',1)
    hold off
    ylabel('Cumulative Effect on FFR');
    title('Federal Funds Rate')
    legend('R&R shocks','VAR shocks','Hybrid VAR shocks')
     



