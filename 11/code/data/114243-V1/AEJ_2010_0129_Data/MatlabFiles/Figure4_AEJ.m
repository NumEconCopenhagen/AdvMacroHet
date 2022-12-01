% counterfactuals.m

% this file estimates IRF's a la R&R, then constructs cunterfactuals s.t.
% IRF of FFR is similar across specifications.

clear all
display(' Output of estimation.m ')

%%% Estimation options
pmeasureVAR=1;                      % select price measure to include in baseline VAR to generate VAR MP shocks: 1 CPI, 2 CPI less housing, 3 PPI, 4 annual CPI inflation, 5 annual CPIc inflation, 6 annual PPI inflation
pmeasureIRF=1;                      % select price measure to include in IRF's: 1 CPI, 2 CPI less housing, 3 PPI, 4 annual CPI inflation, 5 annual CPIc inflation, 6 annual PPI inflation
nirf=60;
options.irfhor=nirf; options.vdechor=nirf; options.constant=1;


%=========================================================================
%    FFR Response: set to one to plot response of FFR to R&R shock
%=========================================================================
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
    
    FFRrespRR=impulseA(FFR2(:,1),[ones(length(uRR2),1) FFR2(:,2:25) uRR2(:,1:37)],nirf,26,1) ;   % extract IRF of FFR to R&R shock.
    
    input1.data=[FFR2(:,1) ones(length(uRR2),1) FFR2(:,2:25) uVAR1(:,1:37)];
    input1.shockpos=26;
    input1.periods=36;
    input1.irf=FFRrespRR(1:input1.periods)';
    input2=input1;
    input2.data=[FFR2(:,1) ones(length(uRR2),1) FFR2(:,2:25) uVAR2(:,1:37)]
    
    options = optimset('MaxFunEvals',1000000);
    options = optimset('MaxIter',10000000,'MaxFunEvals',1000000);
    
    shVAR1 = fminsearch( @(x) shocksearch(x,input1) , [2; zeros(23,1)],options);
    shVAR2 = fminsearch( @(x) shocksearch(x,input2) , [2; zeros(23,1)],options);
    shVAR2 = fminsearch( @(x) shocksearch(x,input2) , shVAR2 ,options);
    shVAR2 = fminsearch( @(x) shocksearch(x,input2) , shVAR2 ,options);
    shVAR1=[shVAR1 ; zeros(nirf-length(shVAR1),1)];    
    shVAR2=[shVAR2 ; zeros(nirf-length(shVAR2),1)];
    
    figure(1)
    f1 = impulseAsh(FFR2(:,1),[ones(length(uRR2),1) FFR2(:,2:25) uVAR1(:,1:37)],nirf,26,1,shVAR1) ; % get impulse response of FFR to VAR shocks that should match IRF of FFR to R&R
    plot(f1)
    hold on
    f2 = impulseAsh(FFR2(:,1),[ones(length(uRR2),1) FFR2(:,2:25) uVAR2(:,1:37)],nirf,26,1,shVAR2) ; % get impulse response of FFR to VAR shocks that should match IRF of FFR to R&R    
    plot(f2,'b--')
    plot(FFRrespRR,'k')
    hold off
    
    %shVAR   = shocksFFR(FFR2(:,1),[ones(length(uRR2),1) FFR2(:,2:25) uVAR2(:,1:37)],nirf,26,FFRrespRR) ;   % get sequence of VAR shocks that yields identical FFR IRF as under R&R
    %shVARrr = shocksFFR(FFR2(:,1),[ones(length(uRR2),1) FFR2(:,2:25) uRR2(:,1:37)],nirf,26,FFRrespRR) ;   % get sequence of VAR_RR shocks that yields identical FFR IRF as under R&R
    
    %f1 = impulseAsh(FFR2(:,1),[ones(length(uRR2),1) FFR2(:,2:25) uVAR1(:,1:37)],nirf,26,1,shVAR) ; % get impulse response of FFR to VAR shocks that should match IRF of FFR to R&R
    
    
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
                
        figure(2)
        subplot(4,3,j)
            if j>6 & j<10
                lag2b=[24 48];
            else
                lag2b=[24 36];
            end
        
        if j<10
                irf=impulse(macro(:,1),[ones(length(uRR2),1) macro(:,2:lag2b(1)+1) sh(:,2:lag2b(2)+1)],nirf,lag2b(1)+2,levelindex) ;            
                sd=se(macro(:,1),[ones(length(uRR2),1) macro(:,2:lag2b(1)+1) sh(:,2:lag2b(2)+1)],nirf,lag2b(1)+2,1000,levelindex) ;
                
        else
                irf=impulseA(macro(:,1),[ones(length(uRR2),1) macro(:,2:lag2b(1)+1) sh(:,1:lag2b(2)+1)],nirf,lag2b(1)+2,levelindex) ;            
                sd=seA(macro(:,1),[ones(length(uRR2),1) macro(:,2:lag2b(1)+1) sh(:,1:lag2b(2)+1)],nirf,lag2b(1)+2,1000,levelindex) ;
        end
        plot(irf,'k','Linewidth',2); 
        hold on
        plot(irf+sd,'b:','Linewidth',2);
        plot(irf-sd,'b:','Linewidth',2);
        if j==1 || j==2 || j==4 || j==5 || j==7 || j==8 || j==10 || j==11       % plot counterfactuals
            if j==1 || j==4 || j==7 || j==10 
                shvar=shVAR1;
            else
                shvar=shVAR2;
            end
            if j<10
                f1 = impulsesh(macro(:,1),[ones(length(uRR2),1) macro(:,2:lag2b(1)+1) sh(:,2:lag2b(2)+1)],nirf,lag2b(1)+2,levelindex,shvar) ; % get impulse response to shock sequence that should match IRF of FFR to R&R shocks
            else
                f1 = impulseAsh(macro(:,1),[ones(length(uRR2),1) macro(:,2:lag2b(1)+1) sh(:,1:lag2b(2)+1)],nirf,lag2b(1)+2,levelindex,shvar) ; % get impulse response to shock sequence that should match IRF of FFR to R&R shocks
            end
            plot(f1,'k--','Linewidth',2)
        end
        hold off
        if j==1 title('VAR Shocks'); ylabel('Output'); elseif j==2 title('VAR with R&R Shocks'); elseif j==3 title('R&R Shocks'); elseif j==4 ylabel('Unemployment'); elseif j==7 ylabel('Price Level'); elseif j==10 ylabel('Effective FFR'); end
    end

            