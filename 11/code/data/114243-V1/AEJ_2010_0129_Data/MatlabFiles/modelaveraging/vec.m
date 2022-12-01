function y = vec(x)

% Syntax:  y = vec(x)
%
% Function version of the vec (:) operator
% Also works on timeseries (returns the data portion of a timeseries).

zzxx1=size(x);
y=[];
for zzkk1=1:zzxx1(1,2)
	y = [y;x(:,zzkk1)];

end

return