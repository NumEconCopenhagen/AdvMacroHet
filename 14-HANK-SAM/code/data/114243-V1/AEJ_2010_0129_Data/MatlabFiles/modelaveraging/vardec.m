% vardec.m

function [ v1, v2, cv1 , cv2, cva , cvb] = vardec(y,x,N,shockpos,levelindex,Tbreak)

% inputs:
% y: dependent variables
% x: matrix of constant, lags of y and lags of shock
% N number of periods to compute variances
% shockpos: position of first shock lag in x
% levelindex: index to determine whether we study the level of variable (1) or changes in variable (0)
% Tbreak: time to break sample for shock variances

ARlags=shockpos-2;
MAlags=cols(x)-1-ARlags;
Beta=inv(x'*x)*(x'*y);                                              % coefficients
ARcoefs=Beta(2:ARlags+1); MAcoefs=Beta(1+ARlags+1:length(Beta));    % AR and MA coefficients
res1=y-x*Beta;                                                       % residuals
mpshocksDAT=[0; x(1:length(x)-1,1+ARlags+1)];         % original shocks (makes use of fact that only lagged shocks are used in regression
maxlags=max(ARlags,MAlags);

for j=1:N
    e1=ceil(Tbreak*rand(1));
    e2=ceil((length(x)-Tbreak)*rand(1));
    
    %resa(j)=res1(e1);
    %resb(j)=res1(e2);
    
    %mpshocks1(j,1)=mpshocksDAT(e1,1);     % this generates MP shocks
    %mpshocks2(j,1)=mpshocksDAT(Tbreak+e2,1);     % this generates MP shocks
    resa(j,1)=std(res1(1:Tbreak))*randn(1);
    resb(j,1)=std(res1(Tbreak+1:length(res1)))*randn(1);
    mpshocks1(j,1)=std(mpshocksDAT(1:Tbreak))*randn(1);
    mpshocks2(j,1)=std(mpshocksDAT(Tbreak+1:length(mpshocksDAT)))*randn(1);
    
end

mpshocks1=[zeros(maxlags,1);mpshocks1]; mpshocks1=makelags(mpshocks1,maxlags);
mpshocks2=[zeros(maxlags,1);mpshocks2]; mpshocks2=makelags(mpshocks2,maxlags);
Xa1=x(1,2:1+ARlags); % initial values
Xa2=x(1,2:1+ARlags); % initial values
Xa3=x(1,2:1+ARlags); % initial values
Xa4=x(1,2:1+ARlags); % initial values

for j=1:N
    Y1(j,1)=Beta(1)+Xa1(j,1:ARlags)*ARcoefs+mpshocks1(j,2:MAlags+1)*MAcoefs;  % first sample
    Y2(j,1)=Beta(1)+Xa2(j,1:ARlags)*ARcoefs+mpshocks2(j,2:MAlags+1)*MAcoefs;  % second sample
    Xa1(j+1,1:ARlags)=[Y1(j,1) Xa1(j,1:ARlags-1)];
    Xa2(j+1,1:ARlags)=[Y2(j,1) Xa2(j,1:ARlags-1)];
    
    Y3(j,1)=Beta(1)+Xa3(j,1:ARlags)*ARcoefs+mpshocks1(j,2:MAlags+1)*MAcoefs+resa(j);  % first sample
    Y4(j,1)=Beta(1)+Xa4(j,1:ARlags)*ARcoefs+mpshocks2(j,2:MAlags+1)*MAcoefs+resb(j);  % second sample
    Xa3(j+1,1:ARlags)=[Y3(j,1) Xa3(j,1:ARlags-1)];
    Xa4(j+1,1:ARlags)=[Y4(j,1) Xa4(j,1:ARlags-1)];
    
end

if levelindex==1
    v1=var(y(1:Tbreak));                % variance of dependent variable, first sample
    v2=var(y(Tbreak+1:length(y)));      % variance of dependent variable, second sample
    cv1=var(Y1);                        % variance conditional on MP shocks, first sample
    cv2=var(Y2);                        % variance conditional on MP shocks, second sample
    cva=var(Y3);
    cvb=var(Y4);
else  % do annual growth rate variances
    v1=var(100*(y(1:Tbreak)-x(1:Tbreak,13)));                         % variance of dependent variable, first sample
    v2=var(100*(y(Tbreak+1:length(y))-x(Tbreak+1:length(y),13)));      % variance of dependent variable, second sample
    cv1=var(100*(Y1(13:N)-Y1(1:N-12)));                        % variance conditional on MP shocks, first sample
    cv2=var(100*(Y2(13:N)-Y2(1:N-12)));                        % variance conditional on MP shocks, second sample
    cva=var(100*(Y3(13:N)-Y3(1:N-12)));                        % variance conditional on MP shocks, first sample
    cvb=var(100*(Y4(13:N)-Y4(1:N-12)));                        % variance conditional on MP shocks, second sample
end
