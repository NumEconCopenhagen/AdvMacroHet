%%% This file runs VAR and spits out Hannan-Quinn criterion.  Allows user to select
%%% smaller number of lags than possible. Note that allows for endogeneous and exogenous variables. 

function hq=hqVAR(data,VARlags,options);

% data is matrix of endogeneous variables ([x(t) x(t-1) .. y(t) y(t-1) ...])
% VARlags is desired number of AR lags to include in VAR 

%%% format data for VAR
T=length(data);
K=cols(data)/(VARlags+1);              % solve for number of variables

 for j=1:K
     Y(:,j)=data(:,(j-1)*(VARlags+1)+1);
     X(:,(j-1)*VARlags+1:(j-1)*VARlags+VARlags) = data(:,(j-1)*(VARlags+1)+2:(j-1)*(VARlags+1)+1+VARlags);
 end
 
 if ~isfield(options,'const')
     options.const=1;               % include a constant term as default
 end
% add constant, time trend, quadratic time trend if required.
 if options.const==1    X=[X ones(length(X),1)]; K1=K+1; elseif options.const==2 X=[X ones(length(X),1) (1:length(X))']; K1=K+2; elseif options.const==3 X=[X ones(length(X),1) (1:length(X))' ((1:length(X)).^2)']; K1=K+3; end
 
 %%% BASELINE ESTIMATES OF VAR
 for j=1:K
    eps(:,j)=Y(:,j)-X*( inv(X'*X)*(X'*Y(:,j)) );                        % residuals
    Beta(:,j)=inv(X'*X)*(X'*Y(:,j));                                    % coefficients
end
Omega=cov(eps);                           % VAR-COV matrix of residuals
 
%  AIC...
logL=-T/2*(K*(1+log(2*3.1416))+log(det(Omega)));
%bic=-2*logL/T+K*((K1-K)+K*VARlags)*log(T)/T;
%aic=-2*logL/T+K*2*((K1-K)+K*VARlags)/T;
hq=-2*logL/T+2*K*((K1-K)+K*VARlags)*log(T)/T;
hq=-hq;

