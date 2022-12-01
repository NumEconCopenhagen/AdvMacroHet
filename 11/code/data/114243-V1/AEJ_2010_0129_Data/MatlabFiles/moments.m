function m = moments(b,Y,X,Z,ivopt)

% this function provides the moments for IV etimation given the
% relationship between Y and b and X specified in ivopt.

if ivopt.linear==1
    e=Y-X*b;
else
    momt = fcnchk(ivopt.momt);
    f = feval(momt,b,Y,X,Z,ivopt);
    e=Y-f;
end

T=length(X);
m = e'*Z/T;
