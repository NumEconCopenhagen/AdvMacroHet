% AICsimVAR.m

% this function takes VAR, simulates data from it, selects 
% optimal lag length via AIC & BIC for each generated sample, and calculates the
% fraction of draws which match pre-specified lag specification.

function [p1, p2, p3]=AICsimVAR(dat,N,lagAIC,lagBIC,VARlag);

% details
% dat is matrix of variables, including VARlag lags of each
% N is # of simulations to do
% lagAIC is lag specification we want to match
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
    VARbmax=0; lagBIC1=[1]; 
    for j=1:36
            % AIC/BIC selection
            dat1=[ip(:,1:j+1) ue(:,1:j+1) p2(:,1:j+1) pcom(:,1:j+1) ffr(:,1:j+1) ]; 
            VARa=aicVAR(dat1,j,options);
            VARb=bicVAR(dat1,j,options);
            if VARa>VARamax   VARamax=VARa;  lagAIC1=j; end
            if VARb>VARbmax   VARbmax=VARb;  lagBIC1=j; end
    end
    
    if lagAIC1==lagAIC
        indic(it,1)=1;
    else
        indic(it,1)=0;
    end
    if lagBIC1==lagBIC
        indic(it,2)=1;
    else
        indic(it,2)=0;
    end
    if lagAIC1==lagAIC & lagBIC1==lagBIC
        indic(it,3)=1;
    else
        indic(it,3)=0;
    end
end

p1=sum(indic(:,1))/N;
p2=sum(indic(:,2))/N;
p3=sum(indic(:,3))/N;
return
    
