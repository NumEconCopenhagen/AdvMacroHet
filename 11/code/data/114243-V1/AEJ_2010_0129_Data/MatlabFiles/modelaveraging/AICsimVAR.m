% AICsimVAR.m

% this function takes VAR, simulates data from it, selects 
% optimal lag length via AIC for each generated sample, and calculates the
% fraction of draws which match pre-specified lag specification.

function [p, P1, P2, P3]=AICsimVAR(dat,N,lagAIC,VARlag,Lags);

% input:
% dat is matrix of variables, including VARlag lags of each
% N is # of simulations to do
% lagAIC is lag specification we want to match
% VARlag is # of lags in the VAR
% Lags is possible VAR lag specifications considered

% output:
% p: probability of observing lagAIC when VARlag is true lag
% P: each P is mean estimated peak effect of MP shock on variable when
% using each of Lags when VARlag is true lag length.  P1 is for IP, P2 for
% UE, P3 for prices.

I=100;
options.constant=1;
out=varcg(dat,VARlag,options);
T=length(dat);
coefs=VARlag;

for it=1:N
    
    % make data
    for j=1:I+T
        e=ceil(length(out.eps)*rand(1));
        shocks(j,:)=out.eps(e,:);
    end
    beta=out.beta; X=zeros(I+T,1+5*coefs);
    X(1,:)=[ dat(1,2:coefs+1) dat(1,coefs+3:2*(coefs+1)) dat(1,2*(coefs+1)+2:3*(coefs+1)) dat(1,3*(coefs+1)+2:4*(coefs+1)) dat(1,4*(coefs+1)+2:5*(coefs+1)) 1];
    Y(1,:)=X(1,:)*beta+shocks(1,:);
    for j=2:I+T
        X(j,5*coefs+1)=1;
        for k=1:5
            X(j,coefs*(k-1)+1:coefs*(k-1)+coefs)=[Y(j-1,k) X(j-1,coefs*(k-1)+1:coefs*(k-1)+coefs-1)];
        end
        Y(j,:)=X(j,:)*beta+shocks(j,:);
    end
    
    % format data for estimation
    ip=Y(I+1-45:I+T,1); ue=Y(I+1-45:I+T,2); p2=Y(I+1-45:I+T,3); pcom=Y(I+1-45:I+T,4); ffr=Y(I+1-45:I+T,5);
    ip=makelags(ip,45); ue=makelags(ue,45); p2=makelags(p2,45); pcom=makelags(pcom,45); ffr=makelags(ffr,45);
    datcoefs=[ip(:,1:coefs+1) ue(:,1:coefs+1) p2(:,1:coefs+1) pcom(:,1:coefs+1) ffr(:,1:coefs+1) ]; % this is specification with correct lags
    outcoefs=varcg(datcoefs,coefs,options);    
    
    VARamax=0; lagAIC1=[1]; 
    
    for j1=1:length(Lags)
        j=Lags(j1);
            % AIC/BIC selection
            dat1=[ip(:,1:j+1) ue(:,1:j+1) p2(:,1:j+1) pcom(:,1:j+1) ffr(:,1:j+1) ]; 
            VARa=aicVAR(dat1,j,options);
            if VARa>VARamax   VARamax=VARa;  lagAIC1=j; end
            
            % estimate peak effects of shocks for each assumed lag specification
            out1=varcg(dat1,j,options);
            P1(it,j1)=min(out1.irf(:,1,5)); % peak effect on IP
            P2(it,j1)=max(out1.irf(:,2,5)); % peak effect on IP
            P3(it,j1)=min(out1.irf(:,3,5)); % peak effect on IP
    end
    
    if lagAIC1==lagAIC
        indic(it,1)=1;
    else
        indic(it,1)=0;
    end
end

p=sum(indic(:,1))/N;
P1=mean(P1);
P2=mean(P2);
P3=mean(P3);
return
    
