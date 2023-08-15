% MAestimation1_open.m

% this file applies the model-averaging procedure to estimate IRF's, saves results.
%clear all
function []=MA3estimation_open_AEJ(var,sample,data)

% parameters
N=1000;              % number of simulations in each MC
periods=60;           % length of IRF's
maxlags=60;           % maximum number of lags set for baseline data
%var=3;                % set to one for IP, 2 for UE, 3 for prices
%sample=1;             % set to one for full sample, 2 to drop 1979-19834.
if var==2 minindex=2; else minindex=1; end % search for min of IRF with IP and P, max for UE

ARlags=[ 12 24 ];  % AR lags considered
MAlags=[ 3 6 9 12 15 18 21 24 27 30 36 39 42 45 ];  % shock lags considered
%if var==3
%    MAlags=[MAlags 45];
%end
ARn=length(ARlags);
MAn=length(MAlags);

%========================================================================
%%%     STEP 1:  data formatting and initial AIC selection
%========================================================================
if data==1
    MPshockdata2;                       % loads data from 1965:1 1996:12
elseif data==2
    MPshockdataGARCH;                   % loads data from GARCH estimation of shocks
elseif data==3
    MPshockdataTVP;                     % loads data from TVP estimation of shocks
elseif data==4
    MPshockdataSW;                      % loads data from Smets-Wouters
    ARlags=ARlags/3; MAlags=MAlags/3; maxlags=maxlags/3; uRR2=uSW; % adjust everything to be quarterly frequency
end
    
IP2=makelags(IP,maxlags); CPI2=makelags(CPI,maxlags); UE2=makelags(UE,maxlags); p2=CPI2; uRR2=makelags(uRR,maxlags);

% truncate sample if needed
if sample==2
    if data<4
    IP2=[IP2(1:(1979+8/12-1969-11/12)*12,:) ; IP2((1984-1969-11/12)*12:length(IP2),:)];
    UE2=[UE2(1:(1979+8/12-1969-11/12)*12,:) ; UE2((1984-1969-11/12)*12:length(UE2),:)];
    p2= [ p2(1:(1979+8/12-1969-11/12)*12,:) ;  p2((1984-1969-11/12)*12:length(p2),:)];
    
    % generate R&R shocks, setting 1979-1982=0, cutting 1979-1984.
    uRR2=uRR;   % shock starts 1965:1
    uRR2((1979+9/12-1964-11/12)*12:(1982+8/12-1964-11/12)*12)=0;
    uRR2=makelags(uRR2,60);
    uRR2=[uRR2(1:(1979+8/12-1969-11/12)*12,:) ; uRR2((1984-1969-11/12)*12:length(uRR2),:)];
    elseif data==4
    IP2=[IP2(1:(1979+2/4-1969-3/4)*4,:) ; IP2((1984-1969-3/4)*4:length(IP2),:)];
    UE2=[UE2(1:(1979+2/4-1969-3/4)*4,:) ; UE2((1984-1969-3/4)*4:length(UE2),:)];
    p2= [ p2(1:(1979+2/4-1969-3/4)*4,:) ;  p2((1984-1969-3/4)*4:length(p2),:)];
    
    % generate R&R shocks, setting 1979-1982=0, cutting 1979-1984.
    uRR2=uSW;   % shock starts 1965:1
    uRR2((1979+3/4-1964-3/4)*4:(1982+2/4-1964-3/4)*4)=0;
    uRR2=makelags(uRR2,20);
    uRR2=[uRR2(1:(1979+2/4-1969-3/4)*4,:) ; uRR2((1984-1969-3/4)*4:length(uRR2),:)];
    end
end
    
% select macro variable for estimation
if var==1                           
    x=IP2(:,1:cols(IP2)-1)-IP2(:,2:cols(IP2)); levelindex=0;
elseif var==2
    x=UE2; levelindex=1;
else
    x=p2(:,1:cols(p2)-1)-p2(:,2:cols(p2)); levelindex=0;
end

% Lag length selection
a1max=-1000; b1max=-1000;
for j1=1:ARn
    j=ARlags(j1);
    for k1=1:MAn
        k=MAlags(k1);
            % AIC selection
            a1=aic(x(:,1),[ones(length(x),1) x(:,2:1+j) uRR2(:,2:1+k)]);
            if a1>a1max   a1max=a1; lagAIC=[j k]; end
            % BIC selection
            b1=bic(x(:,1),[ones(length(x),1) x(:,2:1+j) uRR2(:,2:1+k)]);
            if b1>b1max   b1max=b1; lagBIC=[j k]; end
    end
end

% estimate IRF's at baseline lags and information criteria lags
irfAIC=impulse(x(:,1),[ones(length(x),1) x(:,2:1+lagAIC(1)) uRR2(:,2:1+lagAIC(2))],periods,lagAIC(1)+2,levelindex); % irf of AIC specification
irfBIC=impulse(x(:,1),[ones(length(x),1) x(:,2:1+lagBIC(1)) uRR2(:,2:1+lagBIC(2))],periods,lagBIC(1)+2,levelindex); % irf of BIC specification
if var<3
    if data<4
        irfRR=impulse(x(:,1),[ones(length(x),1) x(:,2:1+24) uRR2(:,2:1+36)],periods,24+2,levelindex); % Baseline R&R for IP and UE
    elseif data==4
        irfRR=impulse(x(:,1),[ones(length(x),1) x(:,2:1+8) uRR2(:,2:1+12)],periods,8+2,levelindex); % Baseline R&R for IP and UE
    end
else
    if data<4
        irfRR=impulse(x(:,1),[ones(length(x),1) x(:,2:1+24) uRR2(:,2:1+45)],periods,24+2,levelindex); % Baseline R&R for P
    elseif data==4
        irfRR=impulse(x(:,1),[ones(length(x),1) x(:,2:1+8) uRR2(:,2:1+15)],periods,8+2,levelindex); % Baseline R&R for P
    end
end

% step 1 complete: lagAIC (lagBIC) is optimal AIC (BIC) lag specification.


%========================================================================
%%%     STEP 2:  iterate through lag specifications and calculate fraction
%%%     of draws yielding same AIC as in data.
%========================================================================
ind=1; 
for j1=1:ARn
    j=ARlags(j1);
    %display(['AR lag: ' num2str(j)])
    for k1=1:MAn
        k=MAlags(k1);
        [p1(j,k) ]=AICsimMA(x(:,1),[ones(length(x),1) x(:,2:1+j) uRR2(:,2:1+k)],N,lagAIC,j,levelindex,periods,ARlags,MAlags,minindex);  
        pa1(ind)=p1(j,k); 
        irf(:,ind)=impulse(x(:,1),[ones(length(x),1) x(:,2:1+j) uRR2(:,2:1+k)],periods,j+2,levelindex);
        ind=ind+1
    end
end
pa1=pa1/sum(pa1);
% step 2 complete


clear IP2 IP UE2 UE CPI2 CPI p2 uRR2 uRR CPIc FFR PCOM PPI PiCPI PiCPIc PiPPI  x
save (['MA3_AEJ' num2str(var) '_' num2str(sample) '_' num2str(data)])

