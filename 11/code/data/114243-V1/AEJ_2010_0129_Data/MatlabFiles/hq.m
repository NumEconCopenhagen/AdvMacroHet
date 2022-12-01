% hq.m

% This file calculates Hannan-Quinn criterion for single equation.

function b = bic(Y,X)

[T,K]=size(X);
beta=inv(X'*X)*(X'*Y);

lf=-(length(Y)/2)*(1+log(2)-log(length(Y)))-length(Y)/2*log((Y-X*beta)'*(Y-X*beta));
b=lf-cols(X)*log(length(Y));

return