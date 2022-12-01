% aic.m

% This file calculates aic criterion for single equation.

function b = aic(Y,X)

%[T,K]=size(X);
beta=inv(X'*X)*(X'*Y);

lf=-(length(Y)/2)*(1+log(2*pi)-log(length(Y)))-length(Y)/2*log((Y-X*beta)'*(Y-X*beta));
b=lf-cols(X);

return