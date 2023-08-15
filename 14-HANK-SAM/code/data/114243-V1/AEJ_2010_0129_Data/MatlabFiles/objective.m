function J = objective(b,Y,X,Z,ivopt)

% this function yields the value of the objective function, given by
% m*(Z'Z)^-1*m', where m are moments.

m = moments(b,Y,X,Z,ivopt);
J = m*inv(Z'*Z)*m';

if ~isfield(ivopt,'min')
    ivopt.min=-inf(size(b));
else
    ivopt.min=ivopt.min+0.001;
end
if ~isfield(ivopt,'max')
    ivopt.max=inf(size(b));
    ivopt.max=ivopt.max-0.001;
end

if b<ivopt.max & b>ivopt.min
    J=J;
else
    J=1e6;
end