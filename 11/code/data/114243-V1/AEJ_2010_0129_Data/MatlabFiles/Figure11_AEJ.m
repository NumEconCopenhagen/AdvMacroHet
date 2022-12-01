% estimationSW.m

% this file provides summary results when using the MP shocks from the
% DSGE model of Smets-Wouters (2007)
clear all
warning('off')

%%% Estimation options and data
options.irfhor=20; options.vdechor=20; options.constant=1;
MPshockdataSW;
IP2=makelags(IP,20); CPI2=makelags(CPI,20);  UE2=makelags(UE,20); p=CPI2;
uRR1=uRR;  % R&R baseline shocks (set equal to uRRg for R&R garch shocks)
uRR2=uSW;  % Smets-Wouters shocks


%======================================================================
%   STEP 1: ESTIMATE SENSITIVITY TO SPECIFIC HISTORICAL EPISODES
%======================================================================

t=1970:1/4:1996.75;
    for j=1:length(IP2)
        % generate shocks set to zero over rolling 3-mo intervals
        sh1=uRR1;   sh2=uRR2;
        sh1(20+j)=0;     sh2(20+j)=0;
        sh1=makelags(sh1,20);  sh2=makelags(sh2,20);
        
        % estimate IRF's using common univariate approach and extract maximal responses
        fitIP=impulse(IP2(:,1)-IP2(:,2),[ones(length(IP2),1) IP2(:,2:9)-IP2(:,3:10) sh1(:,2:13)],20,10,0) ;
        maxIPrr1(j)=min(fitIP);
        fitIP=impulse(IP2(:,1)-IP2(:,2),[ones(length(IP2),1) IP2(:,2:9)-IP2(:,3:10) sh2(:,2:13)],20,10,0) ;
        maxIPrr2(j)=min(fitIP);
                
        fitUE=impulse(UE2(:,1),[ones(length(IP2),1) UE2(:,2:9) sh1(:,2:13)],20,10,1) ;
        maxUErr1(j)=max(fitUE);
        fitUE=impulse(UE2(:,1),[ones(length(IP2),1) UE2(:,2:9) sh2(:,2:13)],20,10,1) ;
        maxUErr2(j)=max(fitUE);
        
        fitP=impulse(p(:,1)-p(:,2),[ones(length(IP2),1) p(:,2:9)-p(:,3:10) sh1(:,2:17)],20,10,0) ;
        maxPrr1(j)=min(fitP);
        fitP=impulse(p(:,1)-p(:,2),[ones(length(IP2),1) p(:,2:9)-p(:,3:10) sh2(:,2:17)],20,10,0) ;
        maxPrr2(j)=min(fitP);
    end

    figure(1)
    subplot(3,3,1)
    plot(t,maxIPrr1,'b:','Linewidth',2)
    hold on
    plot(t,maxIPrr2,'k','Linewidth',2)
    hold off
    title('Sensitivity to Historical Episodes')
    ylabel('Peak Drop in Industrial Production')
    legend('Romer-Romer','Smets-Wouters')

    subplot(3,3,4)
    plot(t,maxUErr1,'b:','Linewidth',2)
    hold on
    plot(t,maxUErr2,'k','Linewidth',2)
    hold off
    ylabel('Peak Rise in Unemployment')
    legend('Romer-Romer','Smets-Wouters')
    
    subplot(3,3,7)
    plot(t,maxPrr1,'b:','Linewidth',2)
    hold on
    plot(t,maxPrr2,'k','Linewidth',2)
    hold off
    ylabel('Peak Drop in Prices')
    legend('Romer-Romer','Smets-Wouters')
    xlabel('Quarter in which shock is set to zero')
    
%=========================================================================
%   STEP 2: ESTIMATE IRF'S FOR WHOLE SAMPLE AND OMITTING 1979-1984
%=========================================================================
    % macro data, dropping 1979-1984: ie data is 1970:1-1978:12 and 1985:1-1996:12
    IP3=[IP2(1:(1979+.5-1969-.75)*4,:) ; IP2((1984-1969-.75)*4:length(IP2),:)];
    UE3=[UE2(1:(1979+.5-1969-.75)*4,:) ; UE2((1984-1969-.75)*4:length(UE2),:)];
    p3= [  p(1:(1979+.5-1969-.75)*4,:) ;   p((1984-1969-.75)*4:length(p),:)];

    % generate R&R GARCH shocks, setting 1979-1982=0, cutting 1979-1984.
    sh3=uRR2;   % shock starts 1965:1
    sh3((1979.75-1964-.75)*4:(1982+.5-1964-.75)*4)=0;
    sh3=makelags(sh3,20);
    sh3=[sh3(1:(1979+.5-1969-.75)*4,:) ; sh3((1984-1969-.75)*4:length(sh3),:)];

    % estimate IRF's for each sample
    irfIP1=impulse(IP2(:,1)-IP2(:,2),[ones(length(IP2),1) IP2(:,2:9)-IP2(:,3:10) sh2(:,2:13)],20,10,0) ;
    irfIP2=impulse(IP3(:,1)-IP3(:,2),[ones(length(IP3),1) IP3(:,2:9)-IP3(:,3:10) sh3(:,2:13)],20,10,0) ;
    irfUE1=impulse(UE2(:,1),[ones(length(IP2),1) UE2(:,2:9) sh2(:,2:13)],20,10,1) ;
    irfUE2=impulse(UE3(:,1),[ones(length(IP3),1) UE3(:,2:9) sh3(:,2:13)],20,10,1) ; 
    irfP1=impulse(p(:,1)-p(:,2),[ones(length(IP2),1) p(:,2:9)-p(:,3:10) sh2(:,2:17)],20,10,0) ;
    irfP2=impulse(p3(:,1)-p3(:,2),[ones(length(IP3),1) p3(:,2:9)-p3(:,3:10) sh3(:,2:17)],20,10,0) ;
    
    % plot results
    subplot(3,3,2)
    plot(irfIP1,'k','Linewidth',2); hold on
    plot(irfIP2,'b--','Linewidth',2); hold off
    title('IRFs using R&R Lags')
    ylabel('Response of Industrial Production')
    legend('Whole sample','Restricted Sample')
    
    subplot(3,3,5)
    plot(irfUE1,'k','Linewidth',2); hold on
    plot(irfUE2,'b--','Linewidth',2); hold off
    ylabel('Response of Unemployment')
    legend('Whole sample','Restricted Sample')            

    subplot(3,3,8)
    plot(irfP1,'k','Linewidth',2); hold on
    plot(irfP2,'b--','Linewidth',2); hold off
    ylabel('Response of Prices')
    legend('Whole sample','Restricted Sample')    
    xlabel('Quarters')
    
%========================================================================
%   STEP 3: VERIFY SENSITIVITY TO LAG LENGTH 
%========================================================================
for j=1:16
    % estimate IRF's with different lags for shocks
    irfIP1=impulse(IP2(:,1)-IP2(:,2),[ones(length(IP2),1) IP2(:,2:9)-IP2(:,3:10) sh2(:,2:j+1)],20,10,0) ;
    irfIP2=impulse(IP3(:,1)-IP3(:,2),[ones(length(IP3),1) IP3(:,2:9)-IP3(:,3:10) sh3(:,2:j+1)],20,10,0) ;
    irfUE1=impulse(UE2(:,1),[ones(length(IP2),1) UE2(:,2:9) sh2(:,2:j+1)],20,10,1) ;
    irfUE2=impulse(UE3(:,1),[ones(length(IP3),1) UE3(:,2:9) sh3(:,2:j+1)],20,10,1) ; 
    irfP1=impulse(p(:,1)-p(:,2),[ones(length(IP2),1) p(:,2:9)-p(:,3:10) sh2(:,2:j+1)],20,10,0) ;
    irfP2=impulse(p3(:,1)-p3(:,2),[ones(length(IP3),1) p3(:,2:9)-p3(:,3:10) sh3(:,2:j+1)],20,10,0) ;
    
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
    plot(maxIP2,'b--','Linewidth',2); hold off
    title('Sensitivity to Lag Length')
    ylabel('Peak Effect on Industrial Production')
    legend('Whole Sample','Restricted Sample')
    
    subplot(3,3,6)
    plot(maxUE1,'k','Linewidth',2); hold on
    plot(maxUE2,'b--','Linewidth',2); hold off
    ylabel('Peak Effect on Unemployment')
    legend('Whole Sample','Restricted Sample')
    
    subplot(3,3,9)
    plot(maxP1,'k','Linewidth',2); hold on
    plot(maxP2,'b--','Linewidth',2); hold off
    ylabel('Peak Effect on Prices')
    legend('Whole Sample','Restricted Sample')
    xlabel('Lags of Shocks Included')
    
