% Figure3_AEJ.m

% This file generates Figure 3 for AEJ paper.

clear all
display(' Output of estimation.m ')

%%% Estimation options
pmeasureVAR=1;                      % select price measure to include in baseline VAR to generate VAR MP shocks: 1 CPI, 2 CPIcore, 3 PPI, 4 annual CPI inflation, 5 annual CPIc inflation, 6 annual PPI inflation
pmeasureIRF=1;                      % select price measure to include in IRF's: 1 CPI, 2 CPIcore, 3 PPI, 4 annual CPI inflation, 5 annual CPIc inflation, 6 annual PPI inflation

%%% What the file runs:
shockindex=1;                       % set to one to plot VAR and R&R shocks together (VAR shock depends on pmeasureVAR, i.e. which measure of prices is included in VAR)


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
    xlim([1969 1997])
    legend('Cumulative VAR MP Shocks','Cumulative R&R MP Shocks','Cumulative R&R VAR Shocks')
    
    figure(2)
    subplot(2,2,1)
    plot(t,sumVAR,'k','Linewidth',2)
    hold on
    plot(t,sumRR,'b:','Linewidth',2)
    xlim([1969 1997])
    hold off
    title('VAR vs R&R Shocks')
    legend('Cumulative VAR Shocks','Cumulative R&R Shocks')
    ylabel('Cumulative Value of Shocks')
    
    subplot(2,2,2)
    plot(t,sumVARrr,'k','Linewidth',2)
    hold on
    plot(t,sumRR,'b:','Linewidth',2)
    hold off
    xlim([1969 1997])
    title('Hybrid VAR vs R&R Shocks')
    legend('Cumulative Hybrid VAR Shocks','Cumulative R&R Shocks')
    ylabel('Cumulative Value of Shocks')

    subplot(2,2,3)
    plot(t,uVAR-uRR,'k','Linewidth',1)
    hold on
    plot(t,zeros(length(t),1),'k','Linewidth',1)
    hold off
    xlim([1969 1997])
    ylabel('Difference between VAR and R&R Shocks')
        
    subplot(2,2,4)
    plot(t,uRRcum-uRR,'k','Linewidth',1)
    hold on
    plot(t,zeros(length(t),1),'k','Linewidth',1)
    hold off
    xlim([1969 1997])
    ylabel('Difference between Hybrid VAR and R&R Shocks')

end

