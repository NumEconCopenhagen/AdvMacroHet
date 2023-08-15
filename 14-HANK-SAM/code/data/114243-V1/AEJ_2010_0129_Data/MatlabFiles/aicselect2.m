% function file that simulates specified equation and calculates average
% lag length selected by AIC, includes
% constant term.

function lagaic = aicselect2(Y,X,N,K)

[T,Z] = size(X)    % T is length of data
J=(Z-1-K);           % number of lags in shocks

beta=inv(X'*X)*(X'*Y);
sigres=std(Y-X*beta);   % std of residuals
sigmp=std(X(:,1+K+1));  % std of mp shocks
beta=beta(2:length(beta)); % drop constant

% simulate data
for it=1:N
    mpshocks=sigmp*randn(100+T,1); mpshocks=[zeros(max(J,K),1);mpshocks]; mpshocks=makelags(mpshocks,max(J,K));
    res=sigres*randn(100+T,1);
    dip=zeros(max(K,J)+1,1);
    dip=makelags(dip,max(K,J));
    for j=1:100+T
        dip(j,1)=dip(j,2:K+1)*beta(1:K)+mpshocks(j,2:J+1)*beta(K+1:length(beta))+res(j);
        dip(j+1,2:K+1)=dip(j,1:K);
    end
        
    % format data for estimation
    dip1=dip(100+1-45:100+T,1);       dip1=makelags(dip1,45);
    mp1=mpshocks(100+1-45:100+T,1);   mp1=makelags(mp1,45);
    
    % do AIC selection given generated data
    a1max=0; lagAIC=[1 1]; ind=1;
    for k=1:36
        for j=1:45
            a1=aic(dip1(:,1),[ones(length(dip1),1) dip1(:,2:1+k) mp1(:,2:1+j)]);
            if a1>a1max   a1max=a1; lagAIC=[k j]; end
        end
    end
    AIC(it,:)=lagAIC;
end

lagaic=mean(AIC);

return