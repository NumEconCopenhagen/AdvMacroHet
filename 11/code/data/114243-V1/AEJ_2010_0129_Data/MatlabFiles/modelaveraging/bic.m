% bic.m

% This file calculates bic criterion for single equation.

function b = bic(Y,X)

[T,K]=size(X);
beta=inv(X'*X)*(X'*Y);

lf=-(length(Y)/2)*(1+log(2*pi)-log(length(Y)))-length(Y)/2*log((Y-X*beta)'*(Y-X*beta));
b=lf-.5*cols(X)*log(length(Y));

return