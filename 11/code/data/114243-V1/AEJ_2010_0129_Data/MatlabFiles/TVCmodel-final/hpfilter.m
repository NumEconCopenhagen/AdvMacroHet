function [gx] = hpfilter(x,lambda,plotter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Kurt Annen annen@web-reg.de
% Date: 15/05/2004
% Internet: www.web-reg.de
%
% The Hodrick Prescott Filter (HP-Filter) is the most popular method to separate
% a time series into its components.
% To accelerate the computation the Add-In makes use of the penta-diagonal structure of
% the coefficient-matrix. So detrending a lot of data points is not a problem for this program
%
% Requires: penta2.m
% x: time series
% lambda= Smoothing parameter (m: 14400, q: 1600, y: 100)
% plot: to plot the result
%
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin < 2
    error('Requires at least two arguments.');
end

[m,n] = size (x);
if m < n
    x = x';
    m = n;
end


a(1)=lambda+1;
a(2)=5*lambda+1;
a(3:m-2)=6*lambda+1;
a(m-1)=5*lambda+1;
a(m)=lambda+1;
b(1)=-2*lambda;
b(2:m-2)=-4*lambda;
b(m-1)=-2*lambda;
c(1:m-2)=lambda;

g=penta2(x,a,b,c);

gx=x-g;

end

