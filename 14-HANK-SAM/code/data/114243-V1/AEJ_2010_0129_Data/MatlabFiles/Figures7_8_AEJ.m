% estimationNoDisinflation.m

% This file estimates IRFS from VAR and R&R over whole sample and over
% sample dropping 1979-1984.  Thus, estimation period is 1970:1-1978:12 and
% 1985:1-1996:12, and shocks between 1979-1982 are set equal to zero.

clear all
display(' Output of estimation.m ')
warning('off')

%%% Estimation options
pmeasureVAR=1;                      % select price measure to include in baseline VAR to generate VAR MP shocks: 1 CPI, 2 CPI less housing, 3 PPI, 4 annual CPI inflation, 5 annual CPIc inflation, 6 annual PPI inflation
pmeasureIRF=1;                      % select price measure to include in IRF's: 1 CPI, 2 CPI less housing, 3 PPI, 4 annual CPI inflation, 5 annual CPIc inflation, 6 annual PPI inflation
options.irfhor=60; options.vdechor=60; options.constant=1;

startbreak=1979+9/12;                % date at which we start cutting sample
endbreak=1984+0/12;                  % date at which sample starts up again
zeroshockend=1982+8/12;              % last date of zero monetary shocks (1982+8/12)

optlag=0;                           % set to one to calculate optimal lag lengths
lagsensitivity=1;                   % set to one to plot sensitivity to lag lengths

    MPshockdata2;                       % loads data from 1965:1 1996:12
    RRcumul;                            % loads R&R cumulative shock (shRR), 1965:1 1996:12
    IP2=makelags(IP,60); CPI2=makelags(CPI,60); CPIc2=makelags(CPIc,60); PPI2=makelags(PPI,60); PiCPI2=makelags(PiCPI,60); PiCPIc2=makelags(PiCPIc,60); PiPPI2=makelags(PiPPI,60); FFR2=makelags(FFR,60); UE2=makelags(UE,60); PCOM2=makelags(PCOM,60); uRR2=makelags(uRR,60); shRR2=makelags(shRR,60); 
    if pmeasureVAR==1 p=CPI2; elseif pmeasureVAR==2 p=CPIc2; elseif pmeasureVAR==3 p=PPI2; elseif pmeasureVAR==4 p=PiCPI2; elseif pmeasureVAR==5 p=PiCPIc2; else p=PiPPI2; end  % select which measure of prices to include in baseline VAR
    
    % baseline VAR measures
    dat2=[IP2(:,1:13) UE2(:,1:13) p(:,1:13) PCOM2(:,1:13) FFR2(:,1:13) ];        
    dat3=[IP2(:,1:13) UE2(:,1:13) p(:,1:13) PCOM2(:,1:13) shRR2(:,1:13)  ];        
    out2=varcg(dat2,12,options);        % basic VAR uses 12 lags (1 year): estimation of VAR and univariate processes start in 1970:1                                             
    out3=varcg(dat3,12,options);        % R&R VAR uses 12 lags (1 year): estimation of VAR and univariate processes start in 1970:1                                             
    
    % generate R&R shocks, setting startbreak-1982=0, cutting startbreak-endbreak.
    uRR3=uRR;   % shock starts 1965:1
    uRR3((startbreak-1964-11/12)*12:(zeroshockend-1964-11/12)*12)=0;
    uRR3=makelags(uRR3,60);
    uRR3=[uRR3(1:(startbreak-1/12-1969-11/12)*12,:) ; uRR3((endbreak-1969-11/12)*12:length(uRR3),:)];
    
    % generate cumulative R&R shocks dropping 1979-1983.
    RRcumA(1)=uRR(1);
    for j=2:(startbreak-1/12-1964-11/12)*12
        RRcumA(j)=RRcumA(j-1)+uRR(j);   % generate cumulative measures for early sample.
    end
    RRcumA=[RRcumA RRcumA(length(RRcumA))*ones(1,(zeroshockend+1/12-startbreak)*12)];    % hold cumulative measure constant between 1979-1982.
    for j=1:(1996+11/12-zeroshockend)*12
        RRcumA=[RRcumA RRcumA(length(RRcumA))+uRR(int16((zeroshockend-1964-11/12)*12+j))]; % restart accumulation in 1983:1.
    end
    RRcumA2=makelags(RRcumA,60);        % this is cumulative shock series, starting in 1970:1, with period between 1979-1982 shocks set to zero.
    
    % macro data, dropping 1979-1984: ie data is 1970:1-1978:12 and 1985:1-1996:12
    IP3=[IP2(1:(startbreak-1/12-1969-11/12)*12,:) ; IP2((endbreak-1969-11/12)*12:length(IP2),:)];
    UE3=[UE2(1:(startbreak-1/12-1969-11/12)*12,:) ; UE2((endbreak-1969-11/12)*12:length(UE2),:)];
    p3= [  p(1:(startbreak-1/12-1969-11/12)*12,:) ;   p((endbreak-1969-11/12)*12:length(p),:)];
    PCOM3=[PCOM2(1:(startbreak-1/12-1969-11/12)*12,:) ; PCOM2((endbreak-1969-11/12)*12:length(PCOM2),:)];
    FFR3=[FFR2(1:(startbreak-1/12-1969-11/12)*12,:) ; FFR2((endbreak-1969-11/12)*12:length(FFR2),:)];
    RRcumA3=[RRcumA2(1:(startbreak-1/12-1969-11/12)*12,:) ; RRcumA2((endbreak-1969-11/12)*12:length(RRcumA2),:)];
    
    % run 2 VAR's dropping period between 1979-1984.
    dat2a=[IP3(:,1:13) UE3(:,1:13) p3(:,1:13) PCOM3(:,1:13) FFR3(:,1:13) ];        
    dat3a=[IP3(:,1:13) UE3(:,1:13) p3(:,1:13) PCOM3(:,1:13) RRcumA3(:,1:13)  ];        
    out2a=varcg(dat2a,12,options);        % basic VAR uses 12 lags (1 year): estimation of VAR and univariate processes start in 1970:1                                             
    out3a=varcg(dat3a,12,options);        % R&R VAR uses 12 lags (1 year): estimation of VAR and univariate processes start in 1970:1                                             
    
    figure('Name','Basic Impulse Responses')
    for j=1:9
        subplot(3,3,j)
        if j==1         plot(out2.irf(1:60,1,5),'k','Linewidth',2'); hold on; plot(out2a.irf(1:60,1,5),'b--','Linewidth',2'); hold off
        elseif j==4     plot(out2.irf(1:60,2,5),'k','Linewidth',2'); hold on; plot(out2a.irf(1:60,2,5),'b--','Linewidth',2'); hold off
        elseif j==7     plot(out2.irf(1:60,3,5),'k','Linewidth',2'); hold on; plot(out2a.irf(1:60,3,5),'b--','Linewidth',2'); hold off
        elseif j==2     plot(out3.irf(1:60,1,5),'k','Linewidth',2'); hold on; plot(out3a.irf(1:60,1,5),'b--','Linewidth',2'); hold off
        elseif j==5     plot(out3.irf(1:60,2,5),'k','Linewidth',2'); hold on; plot(out3a.irf(1:60,2,5),'b--','Linewidth',2'); hold off
        elseif j==8     plot(out3.irf(1:60,3,5),'k','Linewidth',2'); hold on; plot(out3a.irf(1:60,3,5),'b--','Linewidth',2'); hold off;
        end
        if j==3
            irf=impulse(IP2(:,1)-IP2(:,2),[ones(length(uRR2),1) IP2(:,2:25)-IP2(:,3:26) uRR2(:,2:37)],60,26,0) ;
            plot(irf,'k','Linewidth',2); 
            hold on
            irf=impulse(IP3(:,1)-IP3(:,2),[ones(length(uRR3),1) IP3(:,2:25)-IP3(:,3:26) uRR3(:,2:37)],60,26,0) ;
            plot(irf,'b--','Linewidth',2); 
            hold off
        elseif j==6
            irf=impulse(UE2(:,1),[ones(length(uRR2),1) UE2(:,2:25) uRR2(:,2:37)],60,26,1) ;
            plot(irf,'k','Linewidth',2); hold on
            irf=impulse(UE3(:,1),[ones(length(uRR3),1) UE3(:,2:25) uRR3(:,2:37)],60,26,1) ;
            plot(irf,'b--','Linewidth',2); 
            hold off
        elseif j==9
            irf=impulse(p(:,1)-p(:,2),[ones(length(uRR2),1) p(:,2:25)-p(:,3:26) uRR2(:,2:49)],60,26,0) ;
            plot(irf,'k','Linewidth',2); hold on
            irf=impulse(p3(:,1)-p3(:,2),[ones(length(uRR3),1) p3(:,2:25)-p3(:,3:26) uRR3(:,2:49)],60,26,0) ;
            plot(irf,'b--','Linewidth',2); 
            hold off
        end
        if j==1 title('standard VAR Approach'); ylabel('Response of IP'); elseif j==2 title('VAR with R&R shock'); elseif j==3 title('R&R baseline approach'); elseif j==4 ylabel('Response of UE'); elseif j==7 ylabel('Response of CPI'); end
        xlim([0 60]) 
    end

%=========================================================================
%   Optimal Lag length selection for R&R & VAR's
%=========================================================================
if optlag==1
    %MPshockdata2;                       % loads data from 1965:1 1996:12
    %RRcumul;                            % loads R&R cumulative shock (shRR), 1965:1 1996:12
    %IP2=makelags(IP,60); CPI2=makelags(CPI,60); CPIc2=makelags(CPIc,60); PPI2=makelags(PPI,60); PiCPI2=makelags(PiCPI,60); PiCPIc2=makelags(PiCPIc,60); PiPPI2=makelags(PiPPI,60); FFR2=makelags(FFR,60); UE2=makelags(UE,60); PCOM2=makelags(PCOM,60); uRR2=makelags(uRR,60); shRR2=makelags(shRR,60); 
    %if pmeasureVAR==1 p=CPI2; elseif pmeasureVAR==2 p=CPIc2; elseif pmeasureVAR==3 p=PPI2; elseif pmeasureVAR==4 p=PiCPI2; elseif pmeasureVAR==5 p=PiCPIc2; else p=PiPPI2; end  % select which measure of prices to include in baseline VAR
    IP2=IP3; UE2=UE3; p=p3; uRR2=uRR3; PCOM2=PCOM3; FFR2=FFR3; shRR2=RRcumA3;
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
        for j=1:45
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
    
    display('optimal lag lengths via bic for d(IP), UE, and d(p)')
    [lag1b ;  lag2b;   lag3b]
    display('optimal lag lengths via aic for d(IP), UE, and d(p)')
    [lag1a ;  lag2a;   lag3a]
    display('optimal lag lengths via hq for d(IP), UE, and d(p)')
    [lag1c ;  lag2c;   lag3c]
    display('optimal lag lengths for standard VAR: AIC, BIC, HQ')
    [lagVARa lagVARb lagVARc]
    display('optimal lag lengths for R&R VAR: AIC, BIC, HQ')
    [lagRRVARa lagRRVARb lagRRVARc]
    
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
            out1=varcg([IP2(:,1:lagVARa+1) UE2(:,1:lagVARa+1) p(:,1:lagVARa+1) PCOM2(:,1:lagVARa+1) shRR2(:,1:lagVARa+1) ],lagVARa,options);
            plot(out1.irf(:,1,5),'k','Linewidth',2)
            hold on
            out1=varcg([IP2(:,1:lagVARb+1) UE2(:,1:lagVARb+1) p(:,1:lagVARb+1) PCOM2(:,1:lagVARb+1) shRR2(:,1:lagVARb+1) ],lagVARb,options);
            plot(out1.irf(:,1,5),'k--','Linewidth',2)
            out1=varcg([IP2(:,1:lagVARc+1) UE2(:,1:lagVARc+1) p(:,1:lagVARc+1) PCOM2(:,1:lagVARc+1) shRR2(:,1:lagVARc+1) ],lagVARc,options);
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
            out1=varcg([IP2(:,1:lagVARa+1) UE2(:,1:lagVARa+1) p(:,1:lagVARa+1) PCOM2(:,1:lagVARa+1) shRR2(:,1:lagVARa+1) ],lagVARa,options);
            plot(out1.irf(:,2,5),'k','Linewidth',2)
            hold on
            out1=varcg([IP2(:,1:lagVARb+1) UE2(:,1:lagVARb+1) p(:,1:lagVARb+1) PCOM2(:,1:lagVARb+1) shRR2(:,1:lagVARb+1) ],lagVARb,options);
            plot(out1.irf(:,2,5),'k--','Linewidth',2)
            out1=varcg([IP2(:,1:lagVARc+1) UE2(:,1:lagVARc+1) p(:,1:lagVARc+1) PCOM2(:,1:lagVARc+1) shRR2(:,1:lagVARc+1) ],lagVARc,options);
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
            out1=varcg([IP2(:,1:lagVARa+1) UE2(:,1:lagVARa+1) p(:,1:lagVARa+1) PCOM2(:,1:lagVARa+1) shRR2(:,1:lagVARa+1) ],lagVARa,options);
            plot(out1.irf(:,3,5),'k','Linewidth',2)
            hold on
            out1=varcg([IP2(:,1:lagVARb+1) UE2(:,1:lagVARb+1) p(:,1:lagVARb+1) PCOM2(:,1:lagVARb+1) shRR2(:,1:lagVARb+1) ],lagVARb,options);
            plot(out1.irf(:,3,5),'k--','Linewidth',2)
            out1=varcg([IP2(:,1:lagVARc+1) UE2(:,1:lagVARc+1) p(:,1:lagVARc+1) PCOM2(:,1:lagVARc+1) shRR2(:,1:lagVARc+1) ],lagVARc,options);
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
            fit1=fit(IP2(:,1)-IP2(:,2),[ones(length(uRR2),1) IP2(:,2:lag1b(1)+1)-IP2(:,3:lag1b(1)+2) uRR2(:,2:lag1b(2)+1)],60,2+lag1b(1),0) ;
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
            fit1=fit(UE2(:,1),[ones(length(uRR2),1) UE2(:,2:lag2b(1)+1) uRR2(:,2:lag2b(2)+1)],60,lag2b(1)+2,1) ;
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
            fit1=fit(p(:,1)-p(:,2),[ones(length(uRR2),1) p(:,2:lag3b(1)+1)-p(:,3:lag3b(1)+2) uRR2(:,2:lag3b(2)+1)],60,2+lag3b(1),0) ;
            plot(t(13:length(t)),100*(fit1(13:length(fit1))-fit1(1:length(fit1)-12)),'k','Linewidth',2); 
            hold on
            plot(t(13:length(t)),100*(p(13:length(IP2),1)-p(13:length(IP2),13)),'b:','Linewidth',2)
            hold off
        end
        xlim([1970 1997])
        if j==1 title('standard VAR Approach'); ylabel('Annual Growth in IP'); elseif j==2 title('VAR with R&R shock'); elseif j==3 title('R&R baseline approach'); elseif j==4 ylabel('UE Rate'); elseif j==7 ylabel('Annual Growth in CPI'); end
    end        
end

%========================================================================
if lagsensitivity==1
    figure('Name','Lag Sensitivity')
    
    for i=1:6
        if i==1        macro=IP2(:,1:cols(IP2)-1)-IP2(:,2:cols(IP2)); level=0;  uRRa=uRR2;
        elseif i==2    macro=UE2(:,:);                                level=1;  uRRa=uRR2;
        elseif i==3    macro=p(:,1:cols(p)-1)-p(:,2:cols(p));         level=0;  uRRa=uRR2;
        elseif i==4    macro=IP3(:,1:cols(IP2)-1)-IP3(:,2:cols(IP2)); level=0;  uRRa=uRR3;
        elseif i==5    macro=UE3(:,:);                                level=1;  uRRa=uRR3;
        elseif i==6    macro=p3(:,1:cols(p)-1)-p3(:,2:cols(p));       level=0;  uRRa=uRR3;    
        end
    
        if i<4
            ip=IP2; ue=UE2; p1=p; pcom=PCOM2; ffr=FFR2; rrcum=shRR2;
        else
            ip=IP3; ue=UE3; p1=p3; pcom=PCOM3; ffr=FFR3; rrcum=RRcumA3;
        end
        
        
        for j=1:45
            fit1=impulse(macro(:,1),[ones(length(uRRa),1) macro(:,2:25) uRRa(:,2:1+j)],60,26,level) ;
            resp(:,j)=fit1;
            if i==1 || i==3 || i==4 || i==6
                maxRR(j)=min(fit1);
            else
                maxRR(j)=max(fit1);
            end
            lag(j)=j;
        
            dat2a=[ip(:,1:j+1) ue(:,1:j+1) p1(:,1:j+1) pcom(:,1:j+1) ffr(:,1:j+1)  ];        
            out2=varcg(dat2a,j,options);  
            if i==1 || i==4 index=1; elseif i==2 || i==5 index=2; else index=3; end
            if i==1 || i==3 || i==4 || i==6
                maxVAR(j)=min(out2.irf(:,index,5));
            else
                maxVAR(j)=max(out2.irf(:,index,5));
            end
        
            dat3=[ip(:,1:j+1) ue(:,1:j+1) p1(:,1:j+1)  pcom(:,1:j+1) rrcum(:,1:j+1) ];        
            out3=varcg(dat3,j,options);   
            if i==1 || i==4 index=1; elseif i==2 || i==5 index=2; else index=3; end
            if i==1 || i==3 || i==4 || i==6
                maxVARrr(j)=min(out3.irf(:,index,5));
            else
                maxVARrr(j)=max(out3.irf(:,index,5));
            end
        end
    
        subplot(2,3,i)
        plot(lag,maxVAR,'b:','Linewidth',2);
        hold on
        plot(lag,maxVARrr,'b--','Linewidth',2);
        plot(lag,maxRR,'k','Linewidth',2);
        hold off
        xlim([1 48])
        xlabel('Lags in months')
        if i==1 || i==4
            ylabel('Peak Effect of MP shock on Industrial Production')
        elseif i==2 || i==5
            ylabel('Peak Effect of MP shock on Unemployment') 
        else
            ylabel('Peak Effect of MP shock on Prices')
        end
        if i==4
            legend('Baseline VAR','R&R VAR','Baseline R&R')
        end
    end
    
end

sdcs
    %figure('Name','Response of UE for different Lags')
    %h=mesh(ueresp');
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
