% estimation1.m

% This file replicates results for Coibion (2009), comparing the effects of
% MP shocks from VAR to those of R&R (2004), as well as VAR using R&R
% cumulative shock

clear all
display(' Output of estimation.m ')

%%% Estimation options
pmeasureVAR=1;                      % select price measure to include in baseline VAR to generate VAR MP shocks: 1 CPI, 2 CPI less housing, 3 PPI, 4 annual CPI inflation, 5 annual CPIc inflation, 6 annual PPI inflation
pmeasureIRF=1;                      % select price measure to include in IRF's: 1 CPI, 2 CPI less housing, 3 PPI, 4 annual CPI inflation, 5 annual CPIc inflation, 6 annual PPI inflation

%%% What the file runs:
shockindex=0;                       % set to one to plot VAR and R&R shocks together (VAR shock depends on pmeasureVAR, i.e. which measure of prices is included in VAR)
shockcomp=0;                        % set to one to compute different VAR shocks for different measures of price series
comp1=0;                            % set to one to plot impulse responses to MP shocks from VAR and R&R
comp2=0;                            % set to one to plot impulse responses to MP shocks using R&R univariate approach on VAR shocks.
fitindex=1;                         % set to one to plot predicted paths due to MP shocks from VAR and R&R
fitindex1=0;                        % same as fitindex, but feeds R&R shocks into VAR and VAR shocks into R&R
lagsensitivity=0;                   % set to one to assess senstivity of IRF's to lag length
optlag=0;                           % set to one to test for optimal laglength of R&R and VAR.
outliers=0;                         % test for whether R&R IRF's are driven by outlier
FFRresponse=0;                      % set to one to plot response of FFR, IP, UE, and P to each shock, using univariate approach with R&R lags, set to 2 for same using BIC optimal lags.

%%%  Run baseline VAR, extract baseline VAR shocks
MPshockdata;                        % load macro data: IP, UE, CPI, CPIc, PPI, FFR, PCOM: 1968:1-1996:12, monthly
RRshock;                            % load MP shock measure from Romer and Romer (2004): uRR 1969:1-1996:12
IP1=makelags(IP,12); PPI1=makelags(PPI,12);  CPI1=makelags(CPI,12); CPIc1=makelags(CPIc,12); PiPPI1=makelags(PiPPI,12);  PiCPI1=makelags(PiCPI,12); PiCPIc1=makelags(PiCPIc,12); UE1=makelags(UE,12);  FFR1=makelags(FFR,12); PCOM1=makelags(PCOM,12);
if pmeasureVAR==1 p=CPI1; elseif pmeasureVAR==2 p=CPIc1; elseif pmeasureVAR==3 p=PPI1; elseif pmeasureVAR==4 p=PiCPI1; elseif pmeasureVAR==5 p=PiCPIc1; else p=PiPPI1; end  % select which measure of prices to include in baseline VAR
dat1=[IP1 UE1 p  PCOM1 FFR1];      
cumRR(1)=uRR(1);
for i=2:length(uRR)
    cumRR(i)=cumRR(i-1)+uRR(i);
end
cumRR=[zeros(1,12) cumRR];
cumRR1=makelags(cumRR,12);
dat2=[IP1 UE1 p  PCOM1 cumRR1];      
options.irfhor=60; options.vdechor=60; options.constant=1;
out1=varcg(dat1,12,options);        % basic VAR uses 12 lags (1 year)                                             
out2=varcg(dat2,12,options);        % basic VAR uses 12 lags (1 year) with R&R shocks                                            
uVAR=out1.u(:,5);                   % this is baseline MP shock measures from VAR
uRRcum=out2.u(:,5);                 % this is VAR shocks from R&R

%========================================================================
%   Shock Comparison: plot levels and cumulative shock series
%========================================================================
if shockindex==1
    display('Correlation between Shock Measures')
    z=corr(uVAR,uRR)
    t=1969:1/12:1996+11/12;

    figure(1)
    %subplot(2,1,1)
    %plot(t,uVAR,'k','Linewidth',2)
    %hold on
    %plot(t,uRR,'b:','Linewidth',2)
    %hold off
    %xlim([1969 1996])
    %legend('VAR MP Shocks','R&R MP Shocks')

    sumRR(1,1)=uRR(1); sumVAR(1,1)=uVAR(1); sumVARrr(1,1)=uRRcum(1);
    for j=2:length(uRR)
        sumRR(j,1)=sumRR(j-1,1)+uRR(j);
        sumVAR(j,1)=sumVAR(j-1,1)+uVAR(j);
        sumVARrr(j,1)=sumVARrr(j-1,1)+uRRcum(j);        
    end
    display('Correlation between Accumulated Shock Measures')
    z=corr(sumVAR,sumRR)
    %subplot(2,1,2)
    plot(t,sumVAR,'k','Linewidth',2)
    hold on
    plot(t,sumRR,'b:','Linewidth',2)
    plot(t,sumVARrr,'b--','Linewidth',2)
    hold off
    xlim([1969 1996])
    legend('Cumulative VAR MP Shocks','Cumulative R&R MP Shocks','Cumulative R&R VAR Shocks')
    figure(2)
    subplot(2,1,1)
    plot(t,sumVAR,'k','Linewidth',2)
    hold on
    plot(t,sumRR,'b:','Linewidth',2)
    plot(t,sumVARrr,'b--','Linewidth',2)
    hold off
    xlim([1969 1996])
    legend('Cumulative VAR MP Shocks','Cumulative R&R MP Shocks','Cumulative R&R VAR Shocks')
    subplot(2,1,2)
    plot(t,uVAR-uRR,'k','Linewidth',1)
    hold on
    plot(t,uRRcum-uRR,'b','Linewidth',2)
    hold off
    xlim([1969 1996])
    legend('Difference in VAR and R&R Shocks','Difference in R&R VAR and R&R Shocks')
    

end

if shockcomp==1
    for pmeasure=1:6
        if pmeasure==1 p=CPI1; elseif pmeasure==2 p=CPIc1; elseif pmeasure==3 p=PPI1; elseif pmeasure==4 p=PiCPI1; elseif pmeasure==5 p=PiCPIc1; else p=PiPPI1; end  % select which measure of prices to include in baseline VAR
        dat1=[IP1(:,1:13) UE1(:,1:13) p(:,1:13) PCOM1(:,1:13) FFR1(:,1:13)];        
        out1=varcg(dat1,12,options);                                                     
        uVAR1(:,pmeasure)=out1.u(:,5); 
    end
    sumVAR1(1,:)=uVAR1(1,:); sumRR(1,1)=uRR(1); 
    for j=2:length(uRR)
        sumRR(j,1)=sumRR(j-1,1)+uRR(j);
        sumVAR1(j,:)=sumVAR1(j-1,:)+uVAR1(j,:);
    end
    display(' Correlations between Shock Measures ')
    display('    CPI       CPIc       PPI      PiCPI      PiCPIc     PiPPI      R&R')
    corr([uVAR1 uRR])
    display(' Correlations between Accumulated Shock Measures ')
    display('    CPI       CPIc       PPI      PiCPI      PiCPIc     PiPPI      R&R')
    corr([sumVAR1 sumRR])
    t=1969:1/12:1996+11/12;
    figure('Name','Comparison of Cumulative Shocks')
    plot(t,[sumVAR1 sumRR])
    xlim([1969 1996])
    legend('CPI','CPIc','PPI','PiCPI','PiCPIc','PiPPI','R&R')
end
        
%=========================================================================
%   Comparing basic impulse responses from VAR and R&R approach
%=========================================================================
if comp1==1
    MPshockdata2;                       % loads data from 1965:1 1996:12
    RRcumul;                            % loads R&R cumulative shock (shRR), 1965:1 1996:12
    IP2=makelags(IP,60); CPI2=makelags(CPI,60); CPIc2=makelags(CPIc,60); PPI2=makelags(PPI,60); PiCPI2=makelags(PiCPI,60); PiCPIc2=makelags(PiCPIc,60); PiPPI2=makelags(PiPPI,60); FFR2=makelags(FFR,60); UE2=makelags(UE,60); PCOM2=makelags(PCOM,60); uRR2=makelags(uRR,60); shRR2=makelags(shRR,60); 
    if pmeasureVAR==1 p=CPI2; elseif pmeasureVAR==2 p=CPIc2; elseif pmeasureVAR==3 p=PPI2; elseif pmeasureVAR==4 p=PiCPI2; elseif pmeasureVAR==5 p=PiCPIc2; else p=PiPPI2; end  % select which measure of prices to include in baseline VAR
    dat2=[IP2(:,1:13) UE2(:,1:13) p(:,1:13) PCOM2(:,1:13) FFR2(:,1:13) ];        
    dat3=[IP2(:,1:13) UE2(:,1:13) p(:,1:13) PCOM2(:,1:13) shRR2(:,1:13)  ];        
    options.irfstd=1;
    out2=varcg(dat2,12,options);        % basic VAR uses 12 lags (1 year): estimation of VAR and univariate processes start in 1970:1                                             
    out3=varcg(dat3,12,options);        % R&R VAR uses 12 lags (1 year): estimation of VAR and univariate processes start in 1970:1                                             
    
    figure('Name','Basic Impulse Responses')
    for j=1:9
        subplot(3,3,j)
        % plot VAR estimates
        if j==1        
            plot(out2.irf(1:60,1,5),'k','Linewidth',2'); ylim([-0.06 0.02]);
            hold on
            plot(out2.irf(1:60,1,5)+out2.irfstd(1:60,1,5),'b:','Linewidth',2'); 
            plot(out2.irf(1:60,1,5)-out2.irfstd(1:60,1,5),'b:','Linewidth',2'); 
            hold off
        elseif j==4     
            plot(out2.irf(1:60,2,5),'k','Linewidth',2'); ylim([-1 1.5]);
            hold on
            plot(out2.irf(1:60,2,5)+out2.irfstd(1:60,2,5),'b:','Linewidth',2'); 
            plot(out2.irf(1:60,2,5)-out2.irfstd(1:60,2,5),'b:','Linewidth',2'); 
            hold off
        elseif j==7     
            plot(out2.irf(1:60,3,5),'k','Linewidth',2'); ylim([-.06 0.02]);
            hold on
            plot(out2.irf(1:60,3,5)+out2.irfstd(1:60,3,5),'b:','Linewidth',2'); 
            plot(out2.irf(1:60,3,5)-out2.irfstd(1:60,3,5),'b:','Linewidth',2'); 
            hold off
        elseif j==2     
            plot(out3.irf(1:60,1,5),'k','Linewidth',2'); ylim([-0.06 0.02]);
            hold on
            plot(out3.irf(1:60,1,5)+out3.irfstd(1:60,1,5),'b:','Linewidth',2'); 
            plot(out3.irf(1:60,1,5)-out3.irfstd(1:60,1,5),'b:','Linewidth',2'); 
            hold off
        elseif j==5     
            plot(out3.irf(1:60,2,5),'k','Linewidth',2'); ylim([-1 1.5]);
            hold on
            plot(out3.irf(1:60,2,5)+out3.irfstd(1:60,2,5),'b:','Linewidth',2'); 
            plot(out3.irf(1:60,2,5)-out3.irfstd(1:60,2,5),'b:','Linewidth',2'); 
            hold off
        elseif j==8     
            plot(out3.irf(1:60,3,5),'k','Linewidth',2'); ylim([-.06 0.02]);
            hold on
            plot(out3.irf(1:60,3,5)+out3.irfstd(1:60,3,5),'b:','Linewidth',2'); 
            plot(out3.irf(1:60,3,5)-out3.irfstd(1:60,3,5),'b:','Linewidth',2'); 
            hold off
        end
        % plot R&R estimates
        if j==3
            irf=impulse(IP2(:,1)-IP2(:,2),[ones(length(uRR2),1) IP2(:,2:25)-IP2(:,3:26) uRR2(:,2:37)],60,26,0) ;
            plot(irf,'k','Linewidth',2); 
            hold on
            sd=se(IP2(:,1)-IP2(:,2),[ones(length(uRR2),1) IP2(:,2:25)-IP2(:,3:26) uRR2(:,2:37)],60,26,100,0) ;
            plot(irf+sd,'b:','Linewidth',2);
            plot(irf-sd,'b:','Linewidth',2);
            hold off
        elseif j==6
            irf=impulse(UE2(:,1),[ones(length(uRR2),1) UE2(:,2:25) uRR2(:,2:37)],60,26,1) ;
            plot(irf,'k','Linewidth',2); hold on
            sd=se(UE2(:,1),[ones(length(uRR2),1) UE2(:,2:25) uRR2(:,2:37)],60,26,100,1) ;
            plot(irf+sd,'b:','Linewidth',2);
            plot(irf-sd,'b:','Linewidth',2);
            hold off
        elseif j==9
            irf=impulse(p(:,1)-p(:,2),[ones(length(uRR2),1) p(:,2:25)-p(:,3:26) uRR2(:,2:49)],60,26,0) ;
            plot(irf,'k','Linewidth',2); hold on
            sd=se(p(:,1)-p(:,2),[ones(length(uRR2),1) p(:,2:25)-p(:,3:26) uRR2(:,2:49)],60,26,100,0) ;
            plot(irf+sd,'b:','Linewidth',2);
            plot(irf-sd,'b:','Linewidth',2);
            hold off
        end
        if j==1 title('standard VAR Approach'); ylabel('Response of IP'); elseif j==2 title('VAR with R&R shock'); elseif j==3 title('R&R baseline approach'); elseif j==4 ylabel('Response of UE'); elseif j==7 ylabel('Response of CPI'); end
        xlim([0 60]) 
    end
end

%=========================================================================
%   Plot impulse responses from R&R univariate approach using VAR shocks
%=========================================================================
if comp2==1
    MPshockdata2;                       % loads data from 1965:1 1996:12
    IP2=makelags(IP,60); CPI2=makelags(CPI,60); CPIc2=makelags(CPIc,60); PPI2=makelags(PPI,60); PiCPI2=makelags(PiCPI,60); PiCPIc2=makelags(PiCPIc,60); PiPPI2=makelags(PiPPI,60); FFR2=makelags(FFR,60); UE2=makelags(UE,60); PCOM2=makelags(PCOM,60); uRR2=makelags(uRR,60); 
    if pmeasureVAR==1 p=CPI2; elseif pmeasureVAR==2 p=CPIc2; elseif pmeasureVAR==3 p=PPI2; elseif pmeasureVAR==4 p=PiCPI2; elseif pmeasureVAR==5 p=PiCPIc2; else p=PiPPI2; end  % select which measure of prices to include in baseline VAR
    uVAR2=zeros(length(uRR),1);  uVAR2(49:length(uRR),1)=uVAR; uVAR2=makelags(uVAR2,60);
    
    figure('Name','Impulse Responses using R&R methodology')
    for j=1:6
        subplot(3,2,j)
        if j==1         
            irf=impulse(IP2(:,1)-IP2(:,2),[ones(length(uRR2),1) IP2(:,2:25)-IP2(:,3:26) uVAR2(:,2:37)],60,26,0) ;
            plot(irf,'k','Linewidth',2); ylim([-0.05 0.01]);
        elseif j==2
            irf=impulse(IP2(:,1)-IP2(:,2),[ones(length(uRR2),1) IP2(:,2:25)-IP2(:,3:26) uRR2(:,2:37)],60,26,0) ;
            plot(irf,'k','Linewidth',2); ylim([-0.05 0.01]);
        elseif j==3
            irf=impulse(UE2(:,1),[ones(length(uRR2),1) UE2(:,2:25) uVAR2(:,2:37)],60,26,1) ;
            plot(irf,'k','Linewidth',2); ylim([-.05 1]);
        elseif j==4
            irf=impulse(UE2(:,1),[ones(length(uRR2),1) UE2(:,2:25) uRR2(:,2:37)],60,26,1) ;
            plot(irf,'k','Linewidth',2); ylim([-.05 1]);
        elseif j==5
            irf=impulse(p(:,1)-p(:,2),[ones(length(uRR2),1) p(:,2:25)-p(:,3:26) uVAR2(:,2:49)],60,26,0) ;
            plot(irf,'k','Linewidth',2); ylim([-.04 0.01]);
        else
            irf=impulse(p(:,1)-p(:,2),[ones(length(uRR2),1) p(:,2:25)-p(:,3:26) uRR2(:,2:49)],60,26,0) ;
            plot(irf,'k','Linewidth',2); ylim([-.04 0.01]);
        end
         
    end
end

%=========================================================================
%   Compare predicted time paths from VAR and R&R due to MP shocks
%=========================================================================
if fitindex==1
    MPshockdata2;                       % loads data from 1965:1 1996:12
    RRcumul;                            % loads R&R cumulative shock (shRR), 1965:1 1996:12
    IP2=makelags(IP,60); CPI2=makelags(CPI,60); CPIc2=makelags(CPIc,60); PPI2=makelags(PPI,60); PiCPI2=makelags(PiCPI,60); PiCPIc2=makelags(PiCPIc,60); PiPPI2=makelags(PiPPI,60); FFR2=makelags(FFR,60); UE2=makelags(UE,60); PCOM2=makelags(PCOM,60); uRR2=makelags(uRR,60); shRR2=makelags(shRR,60); 
    if pmeasureVAR==1 p=CPI2; elseif pmeasureVAR==2 p=CPIc2; elseif pmeasureVAR==3 p=PPI2; elseif pmeasureVAR==4 p=PiCPI2; elseif pmeasureVAR==5 p=PiCPIc2; else p=PiPPI2; end  % select which measure of prices to include in baseline VAR
    dat2=[IP2(:,1:13) UE2(:,1:13) p(:,1:13) PCOM2(:,1:13) FFR2(:,1:13)  ];        
    dat3=[IP2(:,1:13) UE2(:,1:13) p(:,1:13) PCOM2(:,1:13) shRR2(:,1:13)  ];        
    out2=varcg(dat2,12,options);        % basic VAR uses 12 lags (1 year): estimation of VAR and univariate processes start in 1970:1                                             
    out3=varcg(dat3,12,options);        % R&R VAR uses 12 lags (1 year): estimation of VAR and univariate processes start in 1970:1                                             
    t=1970:1/12:1996+11/12;

    figure('Name','MP-driven fluctuations from VAR and R&R')
    for j=1:9
        subplot(3,3,j)
        if j==1        
            plot(t(13:length(t)),100*(out2.counter(13:length(t),1,5)-out2.counter(1:length(t)-12,1,5)),'k','Linewidth',2'); 
            hold on
            plot(t(13:length(t)),100*(IP2(13:length(IP2),1)-IP2(13:length(IP2),13)),'b:','Linewidth',2)
            R=R2(100*(IP2(13:length(IP2),1)-IP2(13:length(IP2),13)),100*(out2.counter(13:length(t),1,5)-out2.counter(1:length(t)-12,1,5)))
            hold off
        elseif j==2
            plot(t(13:length(t)),100*(out3.counter(13:length(t),1,5)-out3.counter(1:length(t)-12,1,5)),'k','Linewidth',2'); 
            hold on
            plot(t(13:length(t)),100*(IP2(13:length(IP2),1)-IP2(13:length(IP2),13)),'b:','Linewidth',2)
            R=R2(100*(IP2(13:length(IP2),1)-IP2(13:length(IP2),13)),100*(out3.counter(13:length(t),1,5)-out3.counter(1:length(t)-12,1,5)))
            hold off
        elseif j==3
            fit1=fit(IP2(:,1)-IP2(:,2),[ones(length(uRR2),1) IP2(:,2:25)-IP2(:,3:26) uRR2(:,2:37)],60,26,0) ;
            plot(t(13:length(t)),100*(fit1(13:length(fit1))-fit1(1:length(fit1)-12)),'k','Linewidth',2); 
            hold on
            plot(t(13:length(t)),100*(IP2(13:length(IP2),1)-IP2(13:length(IP2),13)),'b:','Linewidth',2)
            R=R2(100*(IP2(13:length(IP2),1)-IP2(13:length(IP2),13)),100*(fit1(13:length(fit1))-fit1(1:length(fit1)-12))')
            hold off
        elseif j==4
            plot(t,out2.counter(:,2,5),'k','Linewidth',2'); 
            hold on
            plot(t,UE2(:,1),'b:','Linewidth',2)
            R=R2(UE2(:,1),out2.counter(:,2,5))
            hold off
        elseif j==5
            plot(t,out3.counter(:,2,5),'k','Linewidth',2'); 
            hold on
            plot(t,UE2(:,1),'b:','Linewidth',2)
            R=R2(UE2(:,1),out3.counter(:,2,5))
            hold off
        elseif j==6
            fit1=fit(UE2(:,1),[ones(length(uRR2),1) UE2(:,2:25) uRR2(:,2:37)],60,26,1) ;
            plot(t,fit1,'k','Linewidth',2); 
            hold on
            plot(t,UE2(:,1),'b:','Linewidth',2)
            R=R2(UE2(:,1),fit1)
            hold off
        elseif j==7
            plot(t(13:length(t)),100*(out2.counter(13:length(t),3,5)-out2.counter(1:length(t)-12,3,5)),'k','Linewidth',2'); 
            hold on
            plot(t(13:length(t)),100*(p(13:length(IP2),1)-p(13:length(IP2),13)),'b:','Linewidth',2)
            R=R2(100*(p(13:length(IP2),1)-p(13:length(IP2),13)),100*(out2.counter(13:length(t),3,5)-out2.counter(1:length(t)-12,3,5)))
            hold off
        elseif j==8
            plot(t(13:length(t)),100*(out3.counter(13:length(t),3,5)-out3.counter(1:length(t)-12,3,5)),'k','Linewidth',2'); 
            hold on
            plot(t(13:length(t)),100*(p(13:length(IP2),1)-p(13:length(IP2),13)),'b:','Linewidth',2)
            R=R2(100*(p(13:length(IP2),1)-p(13:length(IP2),13)),100*(out3.counter(13:length(t),3,5)-out3.counter(1:length(t)-12,3,5)))
            hold off
        elseif j==9
            fit1=fit(p(:,1)-p(:,2),[ones(length(uRR2),1) p(:,2:25)-p(:,3:26) uRR2(:,2:49)],60,26,0) ;
            plot(t(13:length(t)),100*(fit1(13:length(fit1))-fit1(1:length(fit1)-12)),'k','Linewidth',2); 
            hold on
            plot(t(13:length(t)),100*(p(13:length(IP2),1)-p(13:length(IP2),13)),'b:','Linewidth',2)
            R=R2(100*(p(13:length(IP2),1)-p(13:length(IP2),13)),100*(fit1(13:length(fit1))-fit1(1:length(fit1)-12))')
            hold off
        end
        xlim([1970 1997])
        if j==1 title('Standard VAR'); ylabel('Annual Growth in Ind. Prod.'); elseif j==2 title('VAR with R&R Shocks'); elseif j==3 title('R&R Baseline'); elseif j==4 ylabel('Unemployment'); elseif j==7 ylabel('Annual CPI Inflation'); end
    end
end
     
    
   
%=========================================================================
% LAG SENSITIVITY
%=========================================================================
if lagsensitivity==1
    MPshockdata2;                       % loads data from 1965:1 1996:12
    RRcumul;                            % loads R&R cumulative shock (shRR), 1965:1 1996:12
    IP2=makelags(IP,60); CPI2=makelags(CPI,60); CPIc2=makelags(CPIc,60); PPI2=makelags(PPI,60); PiCPI2=makelags(PiCPI,60); PiCPIc2=makelags(PiCPIc,60); PiPPI2=makelags(PiPPI,60); FFR2=makelags(FFR,60); UE2=makelags(UE,60); PCOM2=makelags(PCOM,60); uRR2=makelags(uRR,60); shRR2=makelags(shRR,60); 
    if pmeasureVAR==1 p=CPI2; elseif pmeasureVAR==2 p=CPIc2; elseif pmeasureVAR==3 p=PPI2; elseif pmeasureVAR==4 p=PiCPI2; elseif pmeasureVAR==5 p=PiCPIc2; else p=PiPPI2; end  % select which measure of prices to include in baseline VAR

    t=1970:1/12:1996+11/12;
    fit1=impulse(UE2(:,1),[ones(length(uRR2),1) UE2(:,2:25) uRR2(:,2:37)],60,26,1) ;
    for j=1:48
        fit1=impulse(UE2(:,1),[ones(length(uRR2),1) UE2(:,2:25) uRR2(:,2:1+j)],60,26,1) ;
        ueresp(:,j)=fit1;
        maxueRR(j)=max(fit1);
        lag(j)=j;
        
        fit1=impulse(IP2(:,1)-IP2(:,2),[ones(length(uRR2),1) IP2(:,2:25)-IP2(:,3:26) uRR2(:,2:1+j)],60,26,0) ;
        minipRR(j)=min(fit1);
        
        fit1=impulse(p(:,1)-p(:,2),[ones(length(uRR2),1) p(:,2:25)-p(:,3:26) uRR2(:,2:1+j)],60,26,0) ;
        minpRR(j)=min(fit1);

        %fit1=impulseA(FFR2(:,1),[ones(length(uRR2),1) FFR2(:,2:25) uRR2(:,1:1+j)],60,26,1) ;
        %maxffrRR(j)=max(fit1);
        %t1=find(fit1==maxffrRR(j));
        %t3=find(fit1(t1:length(fit1))<.5*maxffrRR(j));
        %t2=t3(1);
        %hl(j)=t2-t1;
        
        dat2=[IP2(:,1:j+1) UE2(:,1:j+1) p(:,1:j+1) PCOM2(:,1:j+1) FFR2(:,1:j+1)  ];        
        out2=varcg(dat2,j,options);     % basic VAR uses 12 lags (1 year): estimation of VAR and univariate processes start in 1970:1, uses R&R shocks for fit!!!                                            
        maxueVAR(j)=max(out2.irf(:,2,5));
        minipVAR(j)=min(out2.irf(:,1,5));
        minpVAR(j)=min(out2.irf(:,3,5));
        %maxffrVAR(j)=max(out2.irf(:,5,5));
        
        dat3=[IP2(:,1:j+1) UE2(:,1:j+1) p(:,1:j+1)  PCOM2(:,1:j+1) shRR2(:,1:j+1) ];        
        out3=varcg(dat3,j,options);     % basic VAR uses 12 lags (1 year): estimation of VAR and univariate processes start in 1970:1, uses R&R shocks for fit!!!                                            
        maxueRRVAR(j)=max(out3.irf(:,2,5));        
        minipRRVAR(j)=min(out3.irf(:,1,5));
        minpRRVAR(j)=min(out3.irf(:,3,5));
    end
    %figure('Name','Response of UE for different Lags')
    %h=mesh(ueresp');
    figure('Name','Lag Sensitivity')
    subplot(1,3,1)
    plot(lag,minipRR,'k','Linewidth',2)
    hold on
    plot(lag,minipVAR,'b:','Linewidth',2)
    plot(lag,minipRRVAR,'b--','Linewidth',2)
    hold off
    xlim([1 48])
    xlabel('Lag Length in Months')
    ylabel('Peak Effect of MP shock on Industrial Production')
    %legend('R&R Specification','VAR specification','VAR with R&R Shocks')

    subplot(1,3,2)
    plot(lag,maxueRR,'k','Linewidth',2)
    hold on
    plot(lag,maxueVAR,'b:','Linewidth',2)
    plot(lag,maxueRRVAR,'b--','Linewidth',2)
    hold off
    xlim([1 48])
    xlabel('Lag Length in Months')
    ylabel('Peak Effect of MP shock on Unemployment')
    %legend('R&R Specification','VAR specification','VAR with R&R Shocks')

    subplot(1,3,3)
    plot(lag,minpRR,'k','Linewidth',2)
    hold on
    plot(lag,minpVAR,'b:','Linewidth',2)
    plot(lag,minpRRVAR,'b--','Linewidth',2)
    hold off
    xlim([1 48])
    xlabel('Lag Length in Months')
    ylabel('Peak Effect of MP shock on Prices')
    legend('R&R Specification','VAR specification','VAR with R&R Shocks')

    %figure('Name','Max Response of FFR for different lags')
    %plot(lag,hl,'k','Linewidth',2)
    %hold on
    %plot(lag,maxffrVAR,'b:','Linewidth',2)
%    plot(lag,minpRRVAR,'b--','Linewidth',2)
    %hold off
    %xlim([1 48])
    %xlabel('Lag Length in Months')
    %ylabel('Peak Effect of MP shock on FFR')
    %legend('R&R Specification','VAR specification') %,'VAR with R&R Shocks')
    
end

%=========================================================================
%   Optimal Lag length selection for R&R & VAR's
%=========================================================================
if optlag==1
    MPshockdata2;                       % loads data from 1965:1 1996:12
    RRcumul;                            % loads R&R cumulative shock (shRR), 1965:1 1996:12
    IP2=makelags(IP,60); CPI2=makelags(CPI,60); CPIc2=makelags(CPIc,60); PPI2=makelags(PPI,60); PiCPI2=makelags(PiCPI,60); PiCPIc2=makelags(PiCPIc,60); PiPPI2=makelags(PiPPI,60); FFR2=makelags(FFR,60); UE2=makelags(UE,60); PCOM2=makelags(PCOM,60); uRR2=makelags(uRR,60); shRR2=makelags(shRR,60); 
    if pmeasureVAR==1 p=CPI2; elseif pmeasureVAR==2 p=CPIc2; elseif pmeasureVAR==3 p=PPI2; elseif pmeasureVAR==4 p=PiCPI2; elseif pmeasureVAR==5 p=PiCPIc2; else p=PiPPI2; end  % select which measure of prices to include in baseline VAR
    dat2=[IP2(:,1:13) UE2(:,1:13) p(:,1:13) PCOM2(:,1:13) FFR2(:,1:13)  ];        
    b1max=0; b2max=0; b3max=0;
    a1max=0; a2max=0; a3max=0;
    c1max=0; c2max=0; c3max=0;
    VARamax=0; VARbmax=0; VARcmax=0;
    RRVARamax=0; RRVARbmax=0; RRVARcmax=0;
    
    % To do AIC/BIC starting in 1970, do not apply following.  To start in
    % 1974, use following line (eliminates zero values for shocks prior to 1969)
    %IP2=IP2(49:length(IP2),:);     UE2=UE2(49:length(UE2),:);     p=p(49:length(p),:);     uRR2=uRR2(49:length(uRR2),:);    shRR2=shRR2(49:length(shRR2),:);    FFR2=FFR2(49:length(FFR2),:);    PCOM2=PCOM2(49:length(PCOM2),:);    
    
    for k=1:36
        for j=1:48
            % BIC criterion
            b1=bic(IP2(:,1)-IP2(:,2),[ones(length(uRR2),1) IP2(:,2:1+k)-IP2(:,3:2+k) uRR2(:,2:1+j)]);
            b2=bic(UE2(:,1),[ones(length(uRR2),1) UE2(:,2:1+k) uRR2(:,2:1+j)]);
            b3=bic(p(:,1)-p(:,2),[ones(length(uRR2),1) p(:,2:1+k)-p(:,3:2+k) uRR2(:,2:1+j)]);
            if b1>b1max   b1max=b1; lag1b=[k j]; end
            if b2>b2max   b2max=b2; lag2b=[k j]; end
            if b3>b3max   b3max=b3; lag3b=[k j]; end
            
            % AIC criterion
            a1=aic(IP2(:,1)-IP2(:,2),[ones(length(uRR2),1) IP2(:,2:1+k)-IP2(:,3:2+k) uRR2(:,2:1+j)]);
            a2=aic(UE2(:,1),[ones(length(uRR2),1) UE2(:,2:1+k) uRR2(:,2:1+j)]);
            a3=aic(p(:,1)-p(:,2),[ones(length(uRR2),1) p(:,2:1+k)-p(:,3:2+k) uRR2(:,2:1+j)]);
            if a1>a1max   a1max=a1; lag1a=[k j]; end
            if a2>a2max   a2max=a2; lag2a=[k j]; end
            if a3>a3max   a3max=a3; lag3a=[k j]; end

            % HQ criterion
            c1=hq(IP2(:,1)-IP2(:,2),[ones(length(uRR2),1) IP2(:,2:1+k)-IP2(:,3:2+k) uRR2(:,2:1+j)]);
            c2=hq(UE2(:,1),[ones(length(uRR2),1) UE2(:,2:1+k) uRR2(:,2:1+j)]);
            c3=hq(p(:,1)-p(:,2),[ones(length(uRR2),1) p(:,2:1+k)-p(:,3:2+k) uRR2(:,2:1+j)]);
            if c1>c1max   c1max=c1; lag1c=[k j]; end
            if c2>c2max   c2max=c2; lag2c=[k j]; end
            if c3>c3max   c3max=c3; lag3c=[k j]; end
        end
        
        % VAR measure: standard VAR
        VARa=aicVAR([IP2(:,1:k+1) UE2(:,1:k+1) p(:,1:k+1) PCOM2(:,1:k+1) FFR2(:,1:k+1) ],k,options);
        VARb=bicVAR([IP2(:,1:k+1) UE2(:,1:k+1) p(:,1:k+1) PCOM2(:,1:k+1) FFR2(:,1:k+1) ],k,options);
        VARc=hqVAR([IP2(:,1:k+1) UE2(:,1:k+1) p(:,1:k+1)  PCOM2(:,1:k+1) FFR2(:,1:k+1) ],k,options);  
        if VARa>VARamax   VARamax=VARa;  lagVARa=k; end
        if VARb>VARbmax   VARbmax=VARb;  lagVARb=k; end
        if VARc>VARcmax   VARcmax=VARc;  lagVARc=k; end
        
        % VAR measure: R&R shocks
        RRVARa=aicVAR([IP2(:,1:k+1) UE2(:,1:k+1) p(:,1:k+1) PCOM2(:,1:k+1) shRR2(:,1:k+1) ],k,options);
        RRVARb=bicVAR([IP2(:,1:k+1) UE2(:,1:k+1) p(:,1:k+1) PCOM2(:,1:k+1) shRR2(:,1:k+1) ],k,options);
        RRVARc=hqVAR([IP2(:,1:k+1) UE2(:,1:k+1) p(:,1:k+1) PCOM2(:,1:k+1) shRR2(:,1:k+1) ],k,options);  
        if RRVARa>RRVARamax   RRVARamax=RRVARa;  lagRRVARa=k; end
        if RRVARb>RRVARbmax   RRVARbmax=RRVARb;  lagRRVARb=k; end
        if RRVARc>RRVARcmax   RRVARcmax=RRVARc;  lagRRVARc=k; end
        
    end
    
    % calculate peak effects from optimal lag length selection
    % R&R measures
    maxIPrrBAS=min(impulse(IP2(:,1)-IP2(:,2),[ones(length(uRR2),1) IP2(:,2:25)-IP2(:,3:26) uRR2(:,2:37)],60,24+2,0));
    maxIPrrBIC=min(impulse(IP2(:,1)-IP2(:,2),[ones(length(uRR2),1) IP2(:,2:lag1b(1)+1)-IP2(:,3:lag1b(1)+2) uRR2(:,2:lag1b(2)+1)],60,lag1b(1)+2,0));
    maxIPrrAIC=min(impulse(IP2(:,1)-IP2(:,2),[ones(length(uRR2),1) IP2(:,2:lag1a(1)+1)-IP2(:,3:lag1a(1)+2) uRR2(:,2:lag1a(2)+1)],60,lag1a(1)+2,0));
    maxUErrBAS=max(impulse(UE2(:,1),[ones(length(uRR2),1) UE2(:,2:25) uRR2(:,2:37)],60,24+2,1));
    maxUErrAIC=max(impulse(UE2(:,1),[ones(length(uRR2),1) UE2(:,2:lag2a(1)+1) uRR2(:,2:lag2a(2)+1)],60,lag2a(1)+2,1));
    maxUErrBIC=max(impulse(UE2(:,1),[ones(length(uRR2),1) UE2(:,2:lag2b(1)+1) uRR2(:,2:lag2b(2)+1)],60,lag2b(1)+2,1));
    maxPrrBAS=min(impulse(p(:,1)-p(:,2),[ones(length(uRR2),1) p(:,2:25)-p(:,3:26) uRR2(:,2:49)],60,24+2,0));
    maxPrrAIC=min(impulse(p(:,1)-p(:,2),[ones(length(uRR2),1) p(:,2:lag3a(1)+1)-p(:,3:lag3a(1)+2) uRR2(:,2:lag3a(2)+1)],60,lag3a(1)+2,0));
    maxPrrBIC=min(impulse(p(:,1)-p(:,2),[ones(length(uRR2),1) p(:,2:lag3b(1)+1)-p(:,3:lag3b(1)+2) uRR2(:,2:lag3b(2)+1)],60,lag3b(1)+2,0));
    
    %VARs
    outVARbas=varcg([IP2(:,1:12+1) UE2(:,1:12+1) p(:,1:12+1) PCOM2(:,1:12+1) FFR2(:,1:12+1) ],12,options);                               % 12-lag var
    outVARaic=varcg([IP2(:,1:lagVARa+1) UE2(:,1:lagVARa+1) p(:,1:lagVARa+1) PCOM2(:,1:lagVARa+1) FFR2(:,1:lagVARa+1) ],lagVARa,options); % AIC var
    outVARbic=varcg([IP2(:,1:lagVARb+1) UE2(:,1:lagVARb+1) p(:,1:lagVARb+1) PCOM2(:,1:lagVARb+1) FFR2(:,1:lagVARb+1) ],lagVARb,options); % BIC var
    outVARRRbas=varcg([IP2(:,1:12+1) UE2(:,1:12+1) p(:,1:12+1) PCOM2(:,1:12+1) shRR2(:,1:12+1) ],12,options); % AIC var R&R
    outVARRRaic=varcg([IP2(:,1:lagRRVARa+1) UE2(:,1:lagRRVARa+1) p(:,1:lagRRVARa+1) PCOM2(:,1:lagRRVARa+1) shRR2(:,1:lagRRVARa+1) ],lagRRVARa,options); % AIC var R&R
    outVARRRbic=varcg([IP2(:,1:lagRRVARb+1) UE2(:,1:lagRRVARb+1) p(:,1:lagRRVARb+1) PCOM2(:,1:lagRRVARb+1) shRR2(:,1:lagRRVARb+1) ],lagRRVARb,options); % BIC var R&R
    
    display('-------------------------------------------------------')
    display(' Results from Optimal Lag Length Selection, Full Sample')
    display('-------------------------------------------------------')
    display(' ')
    display(' Industrial Production: Lag length then peak effect ')
    display([' VAR Base:  12   ' num2str(min(outVARbas.irf(:,1,5)))])
    display([' VAR AIC:   ' num2str(lagVARa) '  ' num2str(min(outVARaic.irf(:,1,5)))])
    display([' VAR BIC:   ' num2str(lagVARb) '  ' num2str(min(outVARbic.irf(:,1,5)))])
    display([' VARrr Base:  12   ' num2str(min(outVARRRbas.irf(:,1,5)))])
    display([' VARrr AIC: ' num2str(lagRRVARa) '  ' num2str(min(outVARRRaic.irf(:,1,5)))])
    display([' VARrr BIC: ' num2str(lagRRVARb) '  ' num2str(min(outVARRRbic.irf(:,1,5)))])
    display([' R&R Base:  24 36   ' num2str(maxIPrrBAS)])
    display([' R&R AIC:   ' num2str(lag1a) '  ' num2str(maxIPrrAIC)])
    display([' R&R BIC:   ' num2str(lag1b) '  ' num2str(maxIPrrBIC)])
    display(' ')
    
    display(' Unemployment: Lag length then peak effect ')
    display([' VAR Base:  12   ' num2str(max(outVARbas.irf(:,2,5)))])
    display([' VAR AIC:   ' num2str(lagVARa) '  ' num2str(max(outVARaic.irf(:,2,5)))])
    display([' VAR BIC:   ' num2str(lagVARb) '  ' num2str(max(outVARbic.irf(:,2,5)))])
    display([' VARrr Base:  12   ' num2str(max(outVARRRbas.irf(:,2,5)))])
    display([' VARrr AIC: ' num2str(lagRRVARa) '  ' num2str(max(outVARRRaic.irf(:,2,5)))])
    display([' VARrr BIC: ' num2str(lagRRVARb) '  ' num2str(max(outVARRRbic.irf(:,2,5)))])
    display([' R&R Base:  24 36   ' num2str(maxUErrBAS)])
    display([' R&R AIC:   ' num2str(lag2a) '  ' num2str(maxUErrAIC)])
    display([' R&R BIC:   ' num2str(lag2b) '  ' num2str(maxUErrBIC)])
    display(' ')
    
    display(' Prices: Lag length then peak effect ')
    display([' VAR Base:  12   ' num2str(min(outVARbas.irf(:,3,5)))])
    display([' VAR AIC:   ' num2str(lagVARa) '  ' num2str(min(outVARaic.irf(:,3,5)))])
    display([' VAR BIC:   ' num2str(lagVARb) '  ' num2str(min(outVARbic.irf(:,3,5)))])
    display([' VARrr Base:  12   ' num2str(min(outVARRRbas.irf(:,3,5)))])
    display([' VARrr AIC: ' num2str(lagRRVARa) '  ' num2str(min(outVARRRaic.irf(:,3,5)))])
    display([' VARrr BIC: ' num2str(lagRRVARb) '  ' num2str(min(outVARRRbic.irf(:,3,5)))])
    display([' R&R Base:  24 48   ' num2str(maxPrrBAS)])
    display([' R&R AIC:   ' num2str(lag3a) '  ' num2str(maxPrrAIC)])
    display([' R&R BIC:   ' num2str(lag3b) '  ' num2str(maxPrrBIC)])
    display(' ')

    figure('Name','IRFs at optimal lag length')
    for j=1:9
        subplot(3,3,j)
        if j==1
            out1=varcg([IP2(:,1:lagVARa+1) UE2(:,1:lagVARa+1) p(:,1:lagVARa+1) PCOM2(:,1:lagVARa+1) FFR2(:,1:lagVARa+1) ],lagVARa,options);
            plot(out1.irf(:,1,5),'k','Linewidth',2)
            hold on
            out1=varcg([IP2(:,1:lagVARb+1) UE2(:,1:lagVARb+1) p(:,1:lagVARb+1) PCOM2(:,1:lagVARb+1) FFR2(:,1:lagVARb+1) ],lagVARb,options);
            plot(out1.irf(:,1,5),'k--','Linewidth',2)
            out1=varcg([IP2(:,1:lagVARc+1) UE2(:,1:lagVARc+1) p(:,1:lagVARc+1) PCOM2(:,1:lagVARc+1) FFR2(:,1:lagVARc+1) ],lagVARc,options);
            plot(out1.irf(:,1,5),'k:','Linewidth',2)
            hold off
        elseif j==2
            out1=varcg([IP2(:,1:lagRRVARa+1) UE2(:,1:lagRRVARa+1) p(:,1:lagRRVARa+1) PCOM2(:,1:lagRRVARa+1) shRR2(:,1:lagRRVARa+1) ],lagRRVARa,options);
            plot(out1.irf(:,1,5),'k','Linewidth',2)
            hold on
            out1=varcg([IP2(:,1:lagRRVARb+1) UE2(:,1:lagRRVARb+1) p(:,1:lagRRVARb+1) PCOM2(:,1:lagRRVARb+1) shRR2(:,1:lagRRVARb+1) ],lagRRVARb,options);
            plot(out1.irf(:,1,5),'k--','Linewidth',2)
            out1=varcg([IP2(:,1:lagRRVARc+1) UE2(:,1:lagRRVARc+1) p(:,1:lagRRVARc+1) PCOM2(:,1:lagRRVARc+1) shRR2(:,1:lagRRVARc+1) ],lagRRVARc,options);
            plot(out1.irf(:,1,5),'k:','Linewidth',2)
            hold off
        elseif j==3
            irf=impulse(IP2(:,1)-IP2(:,2),[ones(length(uRR2),1) IP2(:,2:lag1a(1)+1)-IP2(:,3:lag1a(1)+2) uRR2(:,2:lag1a(2)+1)],60,lag1a(1)+2,0) ;
            plot(irf,'k','Linewidth',2); hold on
            irf=impulse(IP2(:,1)-IP2(:,2),[ones(length(uRR2),1) IP2(:,2:lag1b(1)+1)-IP2(:,3:lag1b(1)+2) uRR2(:,2:lag1b(2)+1)],60,lag1b(1)+2,0) ;
            plot(irf,'k--','Linewidth',2); 
            irf=impulse(IP2(:,1)-IP2(:,2),[ones(length(uRR2),1) IP2(:,2:lag1c(1)+1)-IP2(:,3:lag1c(1)+2) uRR2(:,2:lag1c(2)+1)],60,lag1c(1)+2,0) ;
            plot(irf,'b-.','Linewidth',2); 
            hold off
        elseif j==4
            out1=varcg([IP2(:,1:lagVARa+1) UE2(:,1:lagVARa+1) p(:,1:lagVARa+1) PCOM2(:,1:lagVARa+1) FFR2(:,1:lagVARa+1) ],lagVARa,options);
            plot(out1.irf(:,2,5),'k','Linewidth',2)
            hold on
            out1=varcg([IP2(:,1:lagVARb+1) UE2(:,1:lagVARb+1) p(:,1:lagVARb+1) PCOM2(:,1:lagVARb+1) FFR2(:,1:lagVARb+1) ],lagVARb,options);
            plot(out1.irf(:,2,5),'k--','Linewidth',2)
            out1=varcg([IP2(:,1:lagVARc+1) UE2(:,1:lagVARc+1) p(:,1:lagVARc+1) PCOM2(:,1:lagVARc+1) FFR2(:,1:lagVARc+1) ],lagVARc,options);
            plot(out1.irf(:,2,5),'k:','Linewidth',2)
            hold off
        elseif j==5
            out1=varcg([IP2(:,1:lagRRVARa+1) UE2(:,1:lagRRVARa+1) p(:,1:lagRRVARa+1) PCOM2(:,1:lagRRVARa+1) shRR2(:,1:lagRRVARa+1) ],lagRRVARa,options);
            plot(out1.irf(:,2,5),'k','Linewidth',2)
            hold on
            out1=varcg([IP2(:,1:lagRRVARb+1) UE2(:,1:lagRRVARb+1) p(:,1:lagRRVARb+1) PCOM2(:,1:lagRRVARb+1) shRR2(:,1:lagRRVARb+1) ],lagRRVARb,options);
            plot(out1.irf(:,2,5),'k--','Linewidth',2)
            out1=varcg([IP2(:,1:lagRRVARc+1) UE2(:,1:lagRRVARc+1) p(:,1:lagRRVARc+1) PCOM2(:,1:lagRRVARc+1) shRR2(:,1:lagRRVARc+1) ],lagRRVARc,options);
            plot(out1.irf(:,2,5),'k:','Linewidth',2)
            hold off
        elseif j==6
            irf=impulse(UE2(:,1),[ones(length(uRR2),1) UE2(:,2:lag2a(1)+1) uRR2(:,2:lag2a(2)+1)],60,lag2a(1)+2,1) ;
            plot(irf,'k','Linewidth',2); hold on
            irf=impulse(UE2(:,1),[ones(length(uRR2),1) UE2(:,2:lag2b(1)+1) uRR2(:,2:lag2b(2)+1)],60,lag2b(1)+2,1) ;
            plot(irf,'k--','Linewidth',2); 
            irf=impulse(UE2(:,1),[ones(length(uRR2),1) UE2(:,2:lag2c(1)+1) uRR2(:,2:lag2c(2)+1)],60,lag2c(1)+2,1) ;
            plot(irf,'k:','Linewidth',2); 
            hold off
        elseif j==7
            out1=varcg([IP2(:,1:lagVARa+1) UE2(:,1:lagVARa+1) p(:,1:lagVARa+1) PCOM2(:,1:lagVARa+1) FFR2(:,1:lagVARa+1) ],lagVARa,options);
            plot(out1.irf(:,3,5),'k','Linewidth',2)
            hold on
            out1=varcg([IP2(:,1:lagVARb+1) UE2(:,1:lagVARb+1) p(:,1:lagVARb+1) PCOM2(:,1:lagVARb+1) FFR2(:,1:lagVARb+1) ],lagVARb,options);
            plot(out1.irf(:,3,5),'k--','Linewidth',2)
            out1=varcg([IP2(:,1:lagVARc+1) UE2(:,1:lagVARc+1) p(:,1:lagVARc+1) PCOM2(:,1:lagVARc+1) FFR2(:,1:lagVARc+1) ],lagVARc,options);
            plot(out1.irf(:,3,5),'k:','Linewidth',2)
            hold off
        elseif j==8
            out1=varcg([IP2(:,1:lagRRVARa+1) UE2(:,1:lagRRVARa+1) p(:,1:lagRRVARa+1) PCOM2(:,1:lagRRVARa+1) shRR2(:,1:lagRRVARa+1) ],lagRRVARa,options);
            plot(out1.irf(:,3,5),'k','Linewidth',2)
            hold on
            out1=varcg([IP2(:,1:lagRRVARb+1) UE2(:,1:lagRRVARb+1) p(:,1:lagRRVARb+1) PCOM2(:,1:lagRRVARb+1) shRR2(:,1:lagRRVARb+1) ],lagRRVARb,options);
            plot(out1.irf(:,3,5),'k--','Linewidth',2)
            out1=varcg([IP2(:,1:lagRRVARc+1) UE2(:,1:lagRRVARc+1) p(:,1:lagRRVARc+1) PCOM2(:,1:lagRRVARc+1) shRR2(:,1:lagRRVARc+1) ],lagRRVARc,options);
            plot(out1.irf(:,3,5),'k:','Linewidth',2)
            hold off
        else
            irf=impulse(p(:,1)-p(:,2),[ones(length(uRR2),1) p(:,2:lag3a(1)+1)-p(:,3:lag3a(1)+2) uRR2(:,2:lag3a(2)+1)],60,lag3a(1)+2,0) ;
            plot(irf,'k','Linewidth',2); hold on
            irf=impulse(p(:,1)-p(:,2),[ones(length(uRR2),1) p(:,2:lag3b(1)+1)-p(:,3:lag3b(1)+2) uRR2(:,2:lag3b(2)+1)],60,lag3b(1)+2,0) ;
            plot(irf,'k--','Linewidth',2); 
            irf=impulse(p(:,1)-p(:,2),[ones(length(uRR2),1) p(:,2:lag3c(1)+1)-p(:,3:lag3c(1)+2) uRR2(:,2:lag3c(2)+1)],60,lag3c(1)+2,0) ;
            plot(irf,'k:','Linewidth',2); 
            hold off
        end

        if j==1 title('Standard VAR'); ylabel('IP'); elseif j==2 title('VAR with R&R Shocks'); elseif j==3 title('R&R Approach'); elseif j==4 ylabel('UE'); elseif j==7 ylabel('Prices'); end
    end
    
    % Plot fitted values using BIC criterion
    figure('Name','Fitted Values from BIC')
    out1=varcg([IP2(:,1:lagVARb+1) UE2(:,1:lagVARb+1) p(:,1:lagVARb+1) PCOM2(:,1:lagVARb+1) FFR2(:,1:lagVARb+1) ],lagVARb,options);              % yields fitted values from standard VAR
    out2=varcg([IP2(:,1:lagRRVARb+1) UE2(:,1:lagRRVARb+1) p(:,1:lagRRVARb+1) PCOM2(:,1:lagRRVARb+1) shRR2(:,1:lagRRVARb+1) ],lagRRVARb,options); % yields fitted values from VAR with R&R shocks
    t=1970:1/12:1996+11/12;
    
    for j=1:9
        subplot(3,3,j)
        if j==1        
            plot(t(13:length(t)),100*(out1.counter(13:length(t),1,5)-out1.counter(1:length(t)-12,1,5)),'k','Linewidth',2'); 
            hold on
            plot(t(13:length(t)),100*(IP2(13:length(IP2),1)-IP2(13:length(IP2),13)),'b:','Linewidth',2)
            hold off
        elseif j==2
            plot(t(13:length(t)),100*(out2.counter(13:length(t),1,5)-out2.counter(1:length(t)-12,1,5)),'k','Linewidth',2'); 
            hold on
            plot(t(13:length(t)),100*(IP2(13:length(IP2),1)-IP2(13:length(IP2),13)),'b:','Linewidth',2)
            hold off
        elseif j==3
            fit1=fit(IP2(:,1)-IP2(:,2),[ones(length(uRR2),1) IP2(:,2:lag1a(1)+1)-IP2(:,3:lag1a(1)+2) uRR2(:,2:lag1a(2)+1)],60,2+lag1a(1),0) ;
            plot(t(13:length(t)),100*(fit1(13:length(fit1))-fit1(1:length(fit1)-12)),'k','Linewidth',2); 
            hold on
            plot(t(13:length(t)),100*(IP2(13:length(IP2),1)-IP2(13:length(IP2),13)),'b:','Linewidth',2)
            hold off
        elseif j==4
            plot(t,out1.counter(:,2,5),'k','Linewidth',2'); 
            hold on
            plot(t,UE2(:,1),'b:','Linewidth',2)
            hold off
        elseif j==5
            plot(t,out2.counter(:,2,5),'k','Linewidth',2'); 
            hold on
            plot(t,UE2(:,1),'b:','Linewidth',2)
            hold off
        elseif j==6
            fit1=fit(UE2(:,1),[ones(length(uRR2),1) UE2(:,2:lag2a(1)+1) uRR2(:,2:lag2a(2)+1)],60,lag2a(1)+2,1) ;
            plot(t,fit1,'k','Linewidth',2); 
            hold on
            plot(t,UE2(:,1),'b:','Linewidth',2)
            hold off
        elseif j==7
            plot(t(13:length(t)),100*(out1.counter(13:length(t),3,5)-out1.counter(1:length(t)-12,3,5)),'k','Linewidth',2'); 
            hold on
            plot(t(13:length(t)),100*(p(13:length(IP2),1)-p(13:length(IP2),13)),'b:','Linewidth',2)
            hold off
        elseif j==8
            plot(t(13:length(t)),100*(out2.counter(13:length(t),3,5)-out2.counter(1:length(t)-12,3,5)),'k','Linewidth',2'); 
            hold on
            plot(t(13:length(t)),100*(p(13:length(IP2),1)-p(13:length(IP2),13)),'b:','Linewidth',2)
            hold off
        elseif j==9
            fit1=fit(p(:,1)-p(:,2),[ones(length(uRR2),1) p(:,2:lag3a(1)+1)-p(:,3:lag3a(1)+2) uRR2(:,2:lag3a(2)+1)],60,2+lag3a(1),0) ;
            plot(t(13:length(t)),100*(fit1(13:length(fit1))-fit1(1:length(fit1)-12)),'k','Linewidth',2); 
            hold on
            plot(t(13:length(t)),100*(p(13:length(IP2),1)-p(13:length(IP2),13)),'b:','Linewidth',2)
            hold off
        end
        xlim([1970 1997])
        if j==1 title('standard VAR Approach'); ylabel('Annual Growth in IP'); elseif j==2 title('VAR with R&R shock'); elseif j==3 title('R&R baseline approach'); elseif j==4 ylabel('UE Rate'); elseif j==7 ylabel('Annual Growth in CPI'); end
    end        
end
   
%end
    
%==========================================================================
%   Outliers: drop one shock at a time, reproduce IRF's of R&R
%==========================================================================
if outliers==1
    MPshockdata2;                       % loads data from 1965:1 1996:12
    IP2=makelags(IP,60); CPI2=makelags(CPI,60); CPIc2=makelags(CPIc,60); PPI2=makelags(PPI,60); PiCPI2=makelags(PiCPI,60); PiCPIc2=makelags(PiCPIc,60); PiPPI2=makelags(PiPPI,60); FFR2=makelags(FFR,60); UE2=makelags(UE,60); PCOM2=makelags(PCOM,60); 
    if pmeasureVAR==1 p=CPI2; elseif pmeasureVAR==2 p=CPIc2; elseif pmeasureVAR==3 p=PPI2; elseif pmeasureVAR==4 p=PiCPI2; elseif pmeasureVAR==5 p=PiCPIc2; else p=PiPPI2; end  % select which measure of prices to include in baseline VAR

    t=1970:1/12:1996+11/12;
    uRR2=makelags(uRR,60); 
    fit1=impulse(UE2(:,1),[ones(length(uRR2),1) UE2(:,2:25) uRR2(:,2:37)],60,26,1) ;  % baseline
    for j=1:length(uRR2)
        uRR1=uRR;
        uRR1(60+j-2:60+j)=0;
        uRR1=makelags(uRR1,60); 
        cumRR1=uRR1(1,:);
        for i=2:length(uRR1)
            cumRR1(i,:)=cumRR1(i-1,:)+uRR1(i,:);
        end
        out1=varcg([IP2(:,1:13) UE2(:,1:13) p(:,1:13) PCOM2(:,1:13) cumRR1(:,1:13)],12,options);
        fitUE=impulse(UE2(:,1),[ones(length(uRR2),1) UE2(:,2:25) uRR1(:,2:37)],60,26,1) ;
        fitIP=impulse(IP2(:,1)-IP2(:,2),[ones(length(uRR2),1) IP2(:,2:25)-IP2(:,3:26) uRR1(:,2:37)],60,26,0) ;
        fitP=impulse(p(:,1)-p(:,2),[ones(length(uRR2),1) p(:,2:25)-p(:,3:26) uRR1(:,2:49)],60,26,0) ;
        fitFFR=impulseA(FFR2(:,1),[ones(length(uRR2),1) FFR2(:,2:25) uRR1(:,1:37)],60,26,1) ;        
        maxUEvar(j)=max(out1.irf(:,2,5));
        maxIPvar(j)=min(out1.irf(:,1,5));
        maxPvar(j)=min(out1.irf(:,3,5));
        %ueresp(:,j)=fit2;
        maxUE(j)=max(fitUE);
        maxIP(j)=min(fitIP);
        maxP(j)=min(fitP);
        maxFFR(j)=max(fitFFR);
        t1=find(fitFFR==maxFFR(j));
        t3=find(fitFFR(t1:length(fitFFR))<.5*maxFFR(j));
        t2=t3(1);
        hl(j)=t2-t1;
    end
    %figure('Name','Response of UE dropping one shock at a time')
    %h=mesh(ueresp');
    figure('Name','Outlier-Sensitivity')
    subplot(2,2,1)
    plot(t,maxIP,'k','Linewidth',2)
    hold on
    plot(t,maxIPvar,'b','Linewidth',1)
    hold off
    title('Industrial Production')
    ylabel('Peak Drop in Industrial Production')
    subplot(2,2,2)
    plot(t,maxUE,'k','Linewidth',2)
%    ylim([-.05 -.01])
    hold on
    plot(t,maxUEvar,'b','Linewidth',1)
    hold off
    title('Unemployment')
    ylabel('Peak Rise in Unemployment')
%    ylim([.3 1.1])
    subplot(2,2,3)
    plot(t,maxP,'k','Linewidth',2)
%    ylim([.3 1.1])
    hold on
     plot(t,maxPvar,'b','Linewidth',1)
     hold off
    title('Prices')
    ylabel('Peak Drop in Prices')
     subplot(2,2,4)
     %plotyy(t,maxFFR,t,hl,'k','Linewidth',2,'k:') %'k','Linewidth',2)
     [AX,H1,H2] = plotyy(t,maxFFR,t,hl,'plot');
     set(H1,'Color','k','Linewidth',2);
        set(H2,'Color','b','LineStyle',':','Linewidth',1);
        set(gca,'YColor','k') ; 
        ylabel('Peak Rise in FFR');
        ylim([1 2.6]);
        axes(AX(2));
        set(gca,'YColor','k')  ;       
        ylabel('Half-Life of FFR Response');
        ylim([0 30]);
%     hold on
 %    ax2 = axes('YAxisLocation','right');
  %   plot(t,hl,'Color','k','Parent',ax2);
   %  hold off
     title('Federal Funds Rate')
     
%    ylim([-.05 -.015])
 end

%=========================================================================
%    FFR Response: set to one to plot response of FFR to R&R shock
%=========================================================================
if FFRresponse>0
    MPshockdata2;                        % load macro data: IP, UE, CPI, CPIc, PPI, FFR, PCOM: 1968:1-1996:12, monthly
    RRcumul;                            % loads R&R cumulative shock (shRR), 1965:1 1996:12
    IP2=makelags(IP,60); CPI2=makelags(CPI,60); CPIc2=makelags(CPIc,60); PPI2=makelags(PPI,60); PiCPI2=makelags(PiCPI,60); PiCPIc2=makelags(PiCPIc,60); PiPPI2=makelags(PiPPI,60); FFR2=makelags(FFR,60); UE2=makelags(UE,60); PCOM2=makelags(PCOM,60); uRR2=makelags(uRR,60); shRR2=makelags(shRR,60); 
    if pmeasureVAR==1 p=CPI2; elseif pmeasureVAR==2 p=CPIc2; elseif pmeasureVAR==3 p=PPI2; elseif pmeasureVAR==4 p=PiCPI2; elseif pmeasureVAR==5 p=PiCPIc2; else p=PiPPI2; end  % select which measure of prices to include in baseline VAR
    dat2=[IP2(:,1:13) UE2(:,1:13) p(:,1:13) PCOM2(:,1:13) FFR2(:,1:13) ];        
    dat3=[IP2(:,1:13) UE2(:,1:13) p(:,1:13) PCOM2(:,1:13) shRR2(:,1:13)  ];        
    out2=varcg(dat2,12,options);        % basic VAR uses 12 lags (1 year): estimation of VAR and univariate processes start in 1970:1                                             
    out3=varcg(dat3,12,options);        % R&R VAR uses 12 lags (1 year): estimation of VAR and univariate processes start in 1970:1                                             
    uVAR1=out2.u(:,5);
    uVAR2=out3.u(:,5);
    uVAR1=[zeros(60,1);uVAR1]; uVAR1=makelags(uVAR1,60);
    uVAR2=[zeros(60,1);uVAR2]; uVAR2=makelags(uVAR2,60);
    
    FFRrespRR=impulseA(FFR2(:,1),[ones(length(uRR2),1) FFR2(:,2:25) uRR2(:,1:37)],60,26,1) ;   % extract IRF of FFR to R&R shock.
    %FFRrespRR(1)=1;
    %for j=2:60
    %    FFRrespRR(j)=.98*FFRrespRR(j-1);
    %end
    shVAR   = shocksFFR(FFR2(:,1),[ones(length(uRR2),1) FFR2(:,2:25) uVAR1(:,1:37)],60,26,FFRrespRR) ;   % get sequence of VAR shocks that yields identical FFR IRF as under R&R
    shVARrr = shocksFFR(FFR2(:,1),[ones(length(uRR2),1) FFR2(:,2:25) uVAR2(:,1:37)],60,26,FFRrespRR) ;   % get sequence of VAR_RR shocks that yields identical FFR IRF as under R&R
    
    %f1 = impulseAsh(FFR2(:,1),[ones(length(uRR2),1) FFR2(:,2:25) uVAR1(:,1:37)],60,26,1,shVAR) ; % get impulse response of FFR to VAR shocks that should match IRF of FFR to R&R
    
    
    for j=1:12
        start=2;
        if j<4      macro=IP2(:,1:60)-IP2(:,2:61); levelindex=0;
        elseif j<7  macro=UE2;                     levelindex=1;
        elseif j<10 macro=p(:,1:60)-p(:,2:61);     levelindex=0;
        else        macro=FFR2;                    levelindex=1; start=1;
        end
        
        if j==1 || j==4 || j==7 || j==10
            sh=uVAR1;
        elseif j==2 || j==5 || j==8 || j==11
            sh=uVAR2;
        else
            sh=uRR2;
        end
                
        figure(1)
        subplot(4,3,j)
        if FFRresponse>1
            for k=1:59
                for i=1:60
                    b2max=-1000; % BIC criterion
                    b2=bic(macro(:,1),[ones(length(uRR2),1) macro(:,2:1+k) sh(:,start:1+i)]);
                    if b2>b2max   b2max=b2; lag2b=[k j]; end
                end
            end
        else
            if j>6 & j<10
                lag2b=[24 48];
            else
                lag2b=[24 36];
            end
        end
        
        if j<10
                irf=impulse(macro(:,1),[ones(length(uRR2),1) macro(:,2:lag2b(1)+1) sh(:,2:lag2b(2)+1)],60,lag2b(1)+2,levelindex) ;            
                sd=se(macro(:,1),[ones(length(uRR2),1) macro(:,2:lag2b(1)+1) sh(:,2:lag2b(2)+1)],60,lag2b(1)+2,100,levelindex) ;
                
        else
                irf=impulseA(macro(:,1),[ones(length(uRR2),1) macro(:,2:lag2b(1)+1) sh(:,1:lag2b(2)+1)],60,lag2b(1)+2,levelindex) ;            
                sd=seA(macro(:,1),[ones(length(uRR2),1) macro(:,2:lag2b(1)+1) sh(:,1:lag2b(2)+1)],60,lag2b(1)+2,100,levelindex) ;
        end
        plot(irf,'k','Linewidth',2); 
        hold on
        plot(irf+sd,'b:','Linewidth',2);
        plot(irf-sd,'b:','Linewidth',2);
        if j==1 || j==2 || j==4 || j==5 || j==7 || j==8 || j==10 || j==11       % plot counterfactuals
            if j==1 || j==4 || j==7 || j==10 
                shvar=shVAR;
            else
                shvar=shVARrr;
            end
            if j<10
                f1 = impulsesh(macro(:,1),[ones(length(uRR2),1) macro(:,2:lag2b(1)+1) sh(:,2:lag2b(2)+1)],60,lag2b(1)+2,levelindex,shvar) ; % get impulse response to shock sequence that should match IRF of FFR to R&R shocks
            else
                f1 = impulseAsh(macro(:,1),[ones(length(uRR2),1) macro(:,2:lag2b(1)+1) sh(:,1:lag2b(2)+1)],60,lag2b(1)+2,levelindex,shvar) ; % get impulse response to shock sequence that should match IRF of FFR to R&R shocks
            end
            plot(f1,'k--','Linewidth',2)
        end
        hold off
        if j==1 title('VAR Shocks'); ylabel('Output'); elseif j==2 title('VAR with R&R Cum. Shock'); elseif j==3 title('R&R Shocks'); elseif j==4 ylabel('Unemployment'); elseif j==7 ylabel('Price Level'); elseif j==10 ylabel('Effective FFR'); end
    end

            
%            subplot(3,1,2)
%    for k=1:60
%        for j=1:60
%            b2max=-1000; % BIC criterion
%            b2=bic(FFR2(:,1),[ones(length(uRR2),1) FFR2(:,2:1+k) uVAR2(:,1:1+j)]);
%            if b2>b2max   b2max=b2; lag2b=[k j]; end
%        end
%    end
%    irf=impulseA(FFR2(:,1),[ones(length(uRR2),1) FFR2(:,2:lag2b(1)+1) uVAR2(:,1:lag2b(2)+1)],60,lag2b(1)+2,1) ;            
%            plot(irf,'k','Linewidth',2); 
%            hold on
%            sd=seA(FFR2(:,1),[ones(length(uRR2),1) FFR2(:,2:lag2b(1)+1) uVAR2(:,1:lag2b(2)+1)],60,lag2b(1)+2,100,1) ;            plot(irf+sd,'b:','Linewidth',2);
%            plot(irf-sd,'b:','Linewidth',2);
%            hold off
%            lag2b
%            display('Cumulative Interest Rate increase VAR2')
%            display([num2str(sum(irf(1:12)')) '  ' num2str(sum(irf(1:24)'))])
%            
%            subplot(3,1,3)
%    for k=1:60
%        for j=1:60
%            b2max=-1000; % BIC criterion
%            b2=bic(FFR2(:,1),[ones(length(uRR2),1) FFR2(:,2:1+k) uRR2(:,1:1+j)]);
%            if b2>b2max   b2max=b2; lag2b=[k j]; end
%        end
%    end
%    irf=impulseA(FFR2(:,1),[ones(length(uRR2),1) FFR2(:,2:lag2b(1)+1) uRR2(:,1:lag2b(2)+1)],60,lag2b(1)+2,1) ;            
%            plot(irf,'k','Linewidth',2); 
%            hold on
%            sd=seA(FFR2(:,1),[ones(length(uRR2),1) FFR2(:,2:lag2b(1)+1) uRR2(:,1:lag2b(2)+1)],60,lag2b(1)+2,100,1) ;            plot(irf+sd,'b:','Linewidth',2);
%            plot(irf-sd,'b:','Linewidth',2);
%            hold off
%            lag2b
%            display('Cumulative Interest Rate increase R&R')
%            display([num2str(sum(irf(1:12)')) '  ' num2str(sum(irf(1:24)'))])
            
%    for k=1:60
%        for j=1:60
%            b2max=-1000; % BIC criterion
%v            b2=hq(FFR2(:,1),[ones(length(uRR2),1) FFR2(:,2:1+k) uRR2(:,1:1+j)]);
%            if b2>b2max   b2max=b2; lag2b=[k j]; end
%        end
%    end
%    irf=impulseA(FFR2(:,1),[ones(length(uRR2),1) FFR2(:,2:lag2b(1)+1) uRR2(:,1:lag2b(2)+1)],60,lag2b(1)+2,1) ;            
%            plot(irf,'k--','Linewidth',2); 
%            hold off
   
end