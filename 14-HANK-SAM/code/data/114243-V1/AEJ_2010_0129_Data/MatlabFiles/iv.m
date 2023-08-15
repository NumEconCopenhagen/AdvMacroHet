function out = iv(b0,ivopt,Y,X,Z)

% This function provides IV estimates of Y on X, given instruments
% Z.  All standard errors are Newey-West HAC, with lags specified in ivopt.
% estimates can be nonlinear.  Function f(b0,X,Z) specifies the
% relationship between coefficients and regressors and must be provided in
% ivopt.momt.  

if ~isfield(ivopt,'linear')
    error('Need to specifiy whether linear')
elseif ~isfield(ivopt,'lags')
    error('Need to specify Newey-West lags in ivopt.lags')
elseif ~isfield(ivopt,'momt') & ivopt.linear==0
    error('Need nonlinear function of coefficients and regressors')
end

if ~isfield(ivopt,'options')
    options=optimset('MaxFunEvals',1000000,'Maxiter',1000000,'TolX',0.00000000001);
else
    options=ivopt.options;
end

nwlags=ivopt.lags;                  % these are lags for Newey-West truncation


%%%% Estimation
if ivopt.linear==1                  % standard linear IV/2SLS estimation
    for j=1:cols(X)
        bhat=inv(Z'*Z)*(Z'*X(:,j));
        Xhat(:,j)=Z*bhat;
    end
    beta=inv(Xhat'*Xhat)*(Xhat'*Y);
    J=0;    flag=1;
else
    [beta, J, flag] = fminsearch('objective',b0,options,Y,X,Z,ivopt);     % find coefficients
end
W=inv(Z'*Z);                        % weighting matrix used to estimate betas
S = NW(beta,Y,X,Z,ivopt);           % get estimate of optimal weighting matrix

% now get var-cov matrix of parameter estimates
T=rows(Y);
inc=0.0000001;                 % increment used in numerical derivative
z=moments(beta,Y,X,Z,ivopt);        % returns moments at estimated parameters
for j=1:length(beta)
    zer=zeros(size(beta));
    zer(j)=inc;
    z1=moments(beta+zer,Y,X,Z,ivopt);  % returns moments at slight change in parameters
    g(:,j)=(z1-z)'/inc;                % construct numerical derivative
end
G=inv(g'*W*g);
V=G*g'*W*S*W*g*G/T;                 % var-cov matrix of estimates

%if ivopt.linear==1
%    V=inv((X'*Z/T)*inv(S)*(Z'*X/T))/T;
%else
%    inc=0.01;                 % increment used in numerical derivative
%    z=moments(beta,Y,X,Z,ivopt);        % returns moments at estimated parameters
%    for j=1:length(beta)
%        zer=zeros(size(beta));
%        zer(j)=inc;
%        z1=moments(beta+zer,Y,X,Z,ivopt);  % returns moments at slight change in parameters
%        g(:,j)=(z1-z)'/inc;                % construct numerical derivative
%    end
%    V=inv(g'*(inv(S))*g)/T;                  % variance-covariance matrix of parameter estimates
%end

% compute standard errors
se=diag(V);
for j=1:length(se)
    se(j)=se(j)^0.5;
end

% compute t-stats
tstat=beta./se;

% compute p-values
for j=1:length(beta)
    pval(j,1)=2*(1-normcdf(abs(tstat(j)),0,1));
end

% output
out.flag=flag;
out.beta=beta;
out.se=se;
out.tstat=beta./se;
out.pval=pval;
out.J=J;
out.betacov=V;
out.S=S;

return







