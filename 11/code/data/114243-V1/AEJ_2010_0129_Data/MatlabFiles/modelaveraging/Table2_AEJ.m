% MAestimation_summary.m

% this file compiles results from MA estimation and calculates all peak
% effects.
clear all
for data=1:1
    if data==1  StoreMat=[]; StoreMatLag=[]; end
    display([' Results for Univariate specification: Data = ' num2str(data)])
    for sample=1:2
        display(['Sample = ' num2str(sample)])
        for var=1:3
            
            load(['MA3_AEJ' num2str(var) '_' num2str(sample) '_' num2str(data)]);
            if var==2
                rrval=max(irfRR);
                aicval=max(irfAIC);
                bicval=max(irfBIC);
                ma1val=max(irf)*pa1';
                %ma2val=max(irf)*wmin;
            else
                rrval=min(irfRR);
                aicval=min(irfAIC);
                bicval=min(irfBIC);
                ma1val=min(irf)*pa1';
                %ma2val=min(irf)*wmin;
            end
            ind=1; Lags=[];
            
            for j1=1:length(ARlags)
                for k1=1:length(MAlags)
                    Lags(ind,:)=[ARlags(j1) MAlags(k1)];
                    ind=ind+1;
                end
            end
            if data==1
                StoreMat=[StoreMat; rrval aicval bicval ma1val];
                StoreMatLag=[ StoreMatLag; 24 36 lagAIC lagBIC pa1*Lags];
            end
            display('R&R     AIC     MA1  ')
            display([num2str(rrval) ' ' num2str(aicval) ' '  num2str(ma1val) ' ' ]) %num2str(ma2val)
            display('Lags for each')
            display([ 'RR  ' num2str(lagAIC) '   '  num2str(pa1*Lags)])
            
        end
        display(' ')
    end
end

for z=1:2
    if z==1
        display([' Results for VAR specification:'])
        load('MA3estimation_VAR_AEJ')
    else
        display([' Results for VAR R&R specification:'])
        load('MA3estimation_VARrr_AEJ')
    end
    display('Results for IP, Whole sample')
    aicval=min(irfaic1(:,1,5));
    bicval=min(irfbic1(:,1,5));
    ma1val=min(irfip1)*pa;
    %ma2val=min(irfip1)*w1(:,1);
    display([num2str(aicval) ' ' num2str(bicval) ' ' num2str(ma1val) ' ' ]) %num2str(ma2val)
    display('Results for IP, Restricted sample')
    aicval=min(irfaic(:,1,5));
    bicval=min(irfbic(:,1,5));
    ma1val=min(irfip)*p;
    %ma2val=min(irfip)*w(:,1);
    display([num2str(aicval) ' ' num2str(bicval) ' ' num2str(ma1val) ' '])% num2str(ma2val)])
    display(' ')
    display('Results for UE, Whole sample')
    aicval=max(irfaic1(:,2,5));
    bicval=max(irfbic1(:,2,5));
    ma1val=max(irfue1)*pa;
    %ma2val=max(irfue1)*w1(:,2);
    display([num2str(aicval) ' ' num2str(bicval) ' ' num2str(ma1val) ' '])% num2str(ma2val)])
    display('Results for UE, Restricted sample')
    aicval=max(irfaic(:,2,5));
    bicval=max(irfbic(:,2,5));
    ma1val=max(irfue)*p;
    %ma2val=max(irfue)*w(:,2);
    display([num2str(aicval) ' ' num2str(bicval) ' ' num2str(ma1val) ' '])% num2str(ma2val)])
    display(' ')            
    display('Results for Prices, Whole sample')
    aicval=min(irfaic1(:,3,5));
    bicval=min(irfbic1(:,3,5));
    ma1val=min(irfp1)*pa;
    %ma2val=min(irfp1)*w1(:,3);
    display([num2str(aicval) ' ' num2str(bicval) ' ' num2str(ma1val) ' '])% num2str(ma2val)])
    display('Results for Prices, Restricted sample')
    aicval=min(irfaic(:,3,5));
    bicval=min(irfbic(:,3,5));
    ma1val=min(irfp)*p;
    %ma2val=min(irfp)*w(:,3);
    display([num2str(aicval) ' ' num2str(bicval) ' ' num2str(ma1val) ' '])% num2str(ma2val)])
    
    display('Lags whole sample')
    display(['RR  ' num2str(lagAIC1) '   ' num2str(lagBIC1) '   ' num2str(Lags*pa)])
    display('Lags restricted sample')
    display(['RR  ' num2str(lagAIC) '   ' num2str(lagBIC) '   ' num2str(Lags*p)])
    display('  ')
end

