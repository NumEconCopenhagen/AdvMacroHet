% This function runs the Kalman filter and smoother
% ALLOW FOR THREE PERIODS WITH DIFFERENT VOLATILITIES:
% a) pre-1979
% b) 1979-1982
% c) post-1982

function [SMstate,SMVstate,KFstate,KFVstate,KFresid,MLE]=KalmanFSbreaks(Y,X,beta,SigmaBB1,R1,SigmaBB2,R2,SigmaBB3,R3)

smplT=length(Y);    % sample size

SMstate=zeros(smplT,length(beta));                  % SMOOTHER (forward run) point estimate in the state (coefs in the Taylor rule)
SMVstate=zeros(smplT,length(beta),length(beta));     % SMOOTHER (backward run) variance of the estimate in the state (coefs in the Taylor rule)
KFstate=zeros(smplT,length(beta));                 % Kalman filter (forward run) point estimate in the state (coefs in the Taylor rule)
KFVstate=zeros(smplT,length(beta),length(beta));    % Kalman filter (forward run)variance of the estimate in the state (coefs in the Taylor rule)
KFresid=zeros(smplT,1);                             % residual (innovation in the Kalman filter)

KFVstate0=zeros(smplT,length(beta),length(beta));    % Kalman filter (forward run)variance of the estimate in the state (coefs in the Taylor rule)

% Notation is based on Lutkepohl (1993) "Introduction to Multiple Time
% Series Analysis", Chapter 13. 

B=eye(length(beta));    % transition matrix for the state (i.e. coefs in the Taylor rule). By assumption, it's a random walk in coefs. 
Sw=SigmaBB1;             % variance of innovation in the state (i.e. coefs in the Taylor rule)
Sv=R1;                   % variance of policy shocks (i.e. error terms in the Taylor rule)


%=========================================================================
%                   Step 1: Kalman filter (forward run)
%=========================================================================
z0=beta;                 % initial value of the state (i.e., OLS estimates of coefs in the Taylor rule on pre-1979 sample)
Sz0=SigmaBB1;            % initial uncertainty about the state (i.e., coefs in the Taylor rule)
    
zt1t1=z0;       
Szt1t1=Sz0;

MLE=0;
for i1=1:smplT
    
    if i1<=130
        Sv=R1;
        Sw=SigmaBB1;  
    end
    if i1>130 & i1<160
        Sv=R2;
        Sw=SigmaBB2;  
    end
    
    if i1>=160 
        Sv=R3;
        Sw=SigmaBB3;  
    end
    
        
    % Prediction step
    zt0t1=B*zt1t1;              % predicted change in the state
    Szt0t1=B*Szt1t1*B'+Sw;      % predicted uncertainty about state
    yt0t1=X(i1,:)*zt0t1;      % predicted value of the FFR
    Syt0t1=X(i1,:)*Szt0t1*X(i1,:)'+Sv;      % predicted uncertainty about FFR

    % correction step
    zt1t1=zt0t1+Szt0t1*X(i1,:)'*inv(X(i1,:)*Szt0t1*X(i1,:)'+Sv)*(Y(i1,1)-yt0t1);
    Szt1t1=Szt0t1-( Szt0t1*X(i1,:)'* inv(X(i1,:)*Szt0t1*X(i1,:)'+Sv) *  X(i1,:)*Szt0t1);
    
    % store estimate state
    KFstate(i1,:)    =zt1t1';
    KFVstate(i1,:,:) =Szt1t1;
    KFVstate0(i1,:,:)=Szt0t1;
    KFresid(i1,1)=Y(i1,1)-yt0t1;
    
    % add to log likelihood
    MLE=MLE-log(det(Syt0t1))/2-(Y(i1,1)-yt0t1)*inv(Syt0t1)*(Y(i1,1)-yt0t1)/2;
end


%=========================================================================
%                   Step 2: Smoother (backward run)
%=========================================================================

zt1T=zt1t1;       
Vt1T=Szt1t1;

SMstate(smplT,:)=zt1T';
SMVstate(smplT,:,:)=Szt1t1;

MLE=0;
for i1=smplT-1:(-1):1
    
    StT=squeeze(KFVstate(i1,:,:))*B'*inv(squeeze(KFVstate0(i1+1,:,:)));
    ztT=KFstate(i1,:)'+StT*(zt1T-B*KFstate(i1,:)');
    VtT=squeeze(KFVstate(i1,:,:))+StT*(Vt1T-squeeze(KFVstate0(i1+1,:,:)))*StT';

    % store estimate state
    SMstate(i1,:)=ztT';
    SMVstate(i1,:,:)=VtT;
    
    Vt1T=VtT;
    zt1T=ztT;

end


