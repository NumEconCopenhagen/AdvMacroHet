% estimationGARCH.m

% this file provides summary results when using the R&R shocks estimated by
% GARCH rather than OLS.
clear all
warning('off')

%%% Estimation options and data
pmeasureVAR=1;  % use CPI for prices (=1)
options.irfhor=60; options.vdechor=60; options.constant=1;
MPshockdata2;
IP2=makelags(IP,60); CPI2=makelags(CPI,60); CPIc2=makelags(CPIc,60); PPI2=makelags(PPI,60); PiCPI2=makelags(PiCPI,60); PiCPIc2=makelags(PiCPIc,60); PiPPI2=makelags(PiPPI,60); FFR2=makelags(FFR,60); UE2=makelags(UE,60); PCOM2=makelags(PCOM,60); 
uRR1=uRR;
if pmeasureVAR==1 p=CPI2; elseif pmeasureVAR==2 p=CPIc2; elseif pmeasureVAR==3 p=PPI2; elseif pmeasureVAR==4 p=PiCPI2; elseif pmeasureVAR==5 p=PiCPIc2; else p=PiPPI2; end  % select which measure of prices to include in baseline VAR
MPshockdata5;                        % loads data with R&R shock estimated by GARCH.
uRR2=uRR;

% note: uRR1 is originial R&R shocks, uRR2 is R&R shocks by GARCH.

%======================================================================
%   STEP 1: ESTIMATE SENSITIVITY TO SPECIFIC HISTORICAL EPISODES
%======================================================================

t=1970:1/12:1996+11/12;
    for j=1:length(IP2)
        % generate shocks set to zero over rolling 3-mo intervals
        sh1=uRR1;   sh2=uRR2;
        sh1(60+j-2:60+j)=0;     sh2(60+j-2:60+j)=0;
        sh1=makelags(sh1,60);  sh2=makelags(sh2,60);
        
        % estimate IRF's using common univariate approach and extract maximal responses
        fitIP=impulse(IP2(:,1)-IP2(:,2),[ones(length(IP2),1) IP2(:,2:25)-IP2(:,3:26) sh1(:,2:37)],60,26,0) ;
        maxIPrr1(j)=min(fitIP);
        fitIP=impulse(IP2(:,1)-IP2(:,2),[ones(length(IP2),1) IP2(:,2:25)-IP2(:,3:26) sh2(:,2:37)],60,26,0) ;
        maxIPrr2(j)=min(fitIP);
                
        fitUE=impulse(UE2(:,1),[ones(length(IP2),1) UE2(:,2:25) sh1(:,2:37)],60,26,1) ;
        maxUErr1(j)=max(fitUE);
        fitUE=impulse(UE2(:,1),[ones(length(IP2),1) UE2(:,2:25) sh2(:,2:37)],60,26,1) ;
        maxUErr2(j)=max(fitUE);
        
        fitP=impulse(p(:,1)-p(:,2),[ones(length(IP2),1) p(:,2:25)-p(:,3:26) sh1(:,2:49)],60,26,0) ;
        maxPrr1(j)=min(fitP);
        fitP=impulse(p(:,1)-p(:,2),[ones(length(IP2),1) p(:,2:25)-p(:,3:26) sh2(:,2:49)],60,26,0) ;
        maxPrr2(j)=min(fitP);
    end

    figure(1)
    subplot(3,3,1)
    plot(t,maxIPrr1,'b:','Linewidth',2)
    hold on
    plot(t,maxIPrr2,'k','Linewidth',1)
    hold off
    title('Sensitivity to Historical Episodes')
    ylabel('Peak Drop in Industrial Production')
    legend('Original R&R shocks','GARCH R&R shocks')

    subplot(3,3,4)
    plot(t,maxUErr1,'b:','Linewidth',2)
    hold on
    plot(t,maxUErr2,'k','Linewidth',1)
    hold off
    ylabel('Peak Rise in Unemployment')
    legend('Original R&R shocks','GARCH R&R shocks')
    
    subplot(3,3,7)
    plot(t,maxPrr1,'b:','Linewidth',2)
    hold on
    plot(t,maxPrr2,'k','Linewidth',1)
    hold off
    ylabel('Peak Drop in Prices')
    legend('Original R&R shocks','GARCH R&R shocks')
    xlabel('Rolling 3-month shocks set to zero')
    
%=========================================================================
%   STEP 2: ESTIMATE IRF'S FOR WHOLE SAMPLE AND OMITTING 1979-1984
%=========================================================================
    % macro data, dropping 1979-1984: ie data is 1970:1-1978:12 and 1985:1-1996:12
    IP3=[IP2(1:(1979+8/12-1969-11/12)*12,:) ; IP2((1984-1969-11/12)*12:length(IP2),:)];
    UE3=[UE2(1:(1979+8/12-1969-11/12)*12,:) ; UE2((1984-1969-11/12)*12:length(UE2),:)];
    p3= [  p(1:(1979+8/12-1969-11/12)*12,:) ;   p((1984-1969-11/12)*12:length(p),:)];

    % generate R&R GARCH shocks, setting 1979-1982=0, cutting 1979-1984.
    sh3=uRR2;   % shock starts 1965:1
    sh3((1979+9/12-1964-11/12)*12:(1982+8/12-1964-11/12)*12)=0;
    sh3=makelags(sh3,60);
    sh3=[sh3(1:(1979+8/12-1969-11/12)*12,:) ; sh3((1984-1969-11/12)*12:length(sh3),:)];

    % estimate IRF's for each sample
    irfIP1=impulse(IP2(:,1)-IP2(:,2),[ones(length(IP2),1) IP2(:,2:25)-IP2(:,3:26) sh2(:,2:37)],60,26,0) ;
    irfIP2=impulse(IP3(:,1)-IP3(:,2),[ones(length(IP3),1) IP3(:,2:25)-IP3(:,3:26) sh3(:,2:37)],60,26,0) ;
    irfUE1=impulse(UE2(:,1),[ones(length(IP2),1) UE2(:,2:25) sh2(:,2:37)],60,26,1) ;
    irfUE2=impulse(UE3(:,1),[ones(length(IP3),1) UE3(:,2:25) sh3(:,2:37)],60,26,1) ; 
    irfP1=impulse(p(:,1)-p(:,2),[ones(length(IP2),1) p(:,2:25)-p(:,3:26) sh2(:,2:49)],60,26,0) ;
    irfP2=impulse(p3(:,1)-p3(:,2),[ones(length(IP3),1) p3(:,2:25)-p3(:,3:26) sh3(:,2:49)],60,26,0) ;
    
    % plot results
    subplot(3,3,2)
    plot(irfIP1,'k','Linewidth',2); hold on
    plot(irfIP2,'b:','Linewidth',2); hold off
    title('IRFs using R&R Lags')
    ylabel('Response of Industrial Production')
    legend('Whole sample','Restricted Sample')
    
    subplot(3,3,5)
    plot(irfUE1,'k','Linewidth',2); hold on
    plot(irfUE2,'b:','Linewidth',2); hold off
    ylabel('Response of Unemployment')
    legend('Whole sample','Restricted Sample')            

    subplot(3,3,8)
    plot(irfP1,'k','Linewidth',2); hold on
    plot(irfP2,'b:','Linewidth',2); hold off
    ylabel('Response of Prices')
    legend('Whole sample','Restricted Sample')    
    xlabel('Months')
    
%========================================================================
%   STEP 3: VERIFY SENSITIVITY TO LAG LENGTH 
%========================================================================
for j=1:45
    % estimate IRF's with different lags for shocks
    irfIP1=impulse(IP2(:,1)-IP2(:,2),[ones(length(IP2),1) IP2(:,2:25)-IP2(:,3:26) sh2(:,2:j+1)],60,26,0) ;
    irfIP2=impulse(IP3(:,1)-IP3(:,2),[ones(length(IP3),1) IP3(:,2:25)-IP3(:,3:26) sh3(:,2:j+1)],60,26,0) ;
    irfUE1=impulse(UE2(:,1),[ones(length(IP2),1) UE2(:,2:25) sh2(:,2:j+1)],60,26,1) ;
    irfUE2=impulse(UE3(:,1),[ones(length(IP3),1) UE3(:,2:25) sh3(:,2:j+1)],60,26,1) ; 
    irfP1=impulse(p(:,1)-p(:,2),[ones(length(IP2),1) p(:,2:25)-p(:,3:26) sh2(:,2:j+1)],60,26,0) ;
    irfP2=impulse(p3(:,1)-p3(:,2),[ones(length(IP3),1) p3(:,2:25)-p3(:,3:26) sh3(:,2:j+1)],60,26,0) ;
    
    % track peak effects for different lags for shocks
    maxIP1(j)=min(irfIP1);
    maxIP2(j)=min(irfIP2);
    maxUE1(j)=max(irfUE1);
    maxUE2(j)=max(irfUE2);
    maxP1(j)=min(irfP1);
    maxP2(j)=min(irfP2);
end

    % plot
    subplot(3,3,3)
    plot(maxIP1,'k','Linewidth',2); hold on
    plot(maxIP2,'b:','Linewidth',1); hold off
    title('Sensitivity to Lag Length')
    ylabel('Peak Effect on Industrial Production')
    legend('Whole Sample','Restricted Sample')
    
    subplot(3,3,6)
    plot(maxUE1,'k','Linewidth',2); hold on
    plot(maxUE2,'b:','Linewidth',1); hold off
    ylabel('Peak Effect on Unemployment')
    legend('Whole Sample','Restricted Sample')
    
    subplot(3,3,9)
    plot(maxP1,'k','Linewidth',2); hold on
    plot(maxP2,'b:','Linewidth',1); hold off
    ylabel('Peak Effect on Prices')
    legend('Whole Sample','Restricted Sample')
    xlabel('Lags of Shocks Included')
    
