% This function computes the Quandt likelihood ratio test for instability
function J=QLR(Y,X)

s0=0.15;    % lower bound on the sample over which to search for a break
s1=0.85;    % lower bound on the sample over which to search for a break

start0=round(length(Y)*s0);
end0=round(length(Y)*s1);

[n1 n2]=size(X);

% fit contrained regression
beta=X\Y;
resid0=Y-X*beta;
V0=resid0'*resid0;

Jmat=[];
% fit unconstrained regression
for t=start0:end0
    dummy0=[zeros(t-1,n2); ones(length(Y)-t+1,n2)];
    Xm=[X X.*dummy0];
    beta=Xm\Y;
    resid1=Y-Xm*beta;
    V1=resid1'*resid1;
    
    % test for the break at period t
    Fstat=((V0-V1)/n2)/(V1/(n1-n2));

    Jmat=[Jmat; Fstat];
end

J=max(Jmat);
end



