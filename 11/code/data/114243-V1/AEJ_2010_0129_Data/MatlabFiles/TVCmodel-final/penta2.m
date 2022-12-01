function [x]= penta2(y,a,b,c)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Kurt Annen annen@web-reg.de
% Date: 15/05/2004
% Internet: www.web-reg.de
%
% Solves the problem Ax=b when A is pentadiagonal and strongly nonsingular. 
% This is much faster than x=A\y for large matrices.  
%
% Reference: Späth, Helmuth "Numerik: Eine Einführung für Mathematiker und Informatiker"
%               S. 110 . Vieweg-Verlag Braunschweig/Wiesbaden (1994)
%
% a = main diagonal
% b = 2. diagonal
% c = 3. diagonal
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin ~= 4
    error('penta(A,y) requires four arguments');
end

[n] = length(a);
[m] = length(y);
[o] = length(b);
[p] = length(c);

if m ~= n
    error('Inner matrix dimensions must agree');
end
c(m)=0;
c(m-1)=0;
b(m)=0;  

if m ~= length(b) &  m ~= length(c)
    error('a,b,c must have the same dimension');
end    


% a optimized algorithm

    h1=0;
    h2=0;
    h3=0;
    h4=0;
    h5=0;
    hh1=0;
    hh2=0;
    hh3=0;
    hh4=0;
    hh5=0;
    z=0;
    hb=0;
    hc=0;
    
    for i=1:m
        z=a(i)-h4*h1-hh5*hh2;
        hb=b(i);
        hh1=h1;
        h1=(hb-h4*h2)/z;
        b(i)=h1;
        hc=c(i);
        hh2=h2;
        h2=hc/z;
        c(i)=h2;
        a(i)=(y(i)-hh3*hh5-h3*h4)/z;
        hh3=h3;
        h3=a(i);
        h4=hb-h5*hh1;
        hh5=h5;
        h5=hc;
    end
    h2=0;
    h1=a(m);
    y(m)=h1;
    for i=m:-1:1
        y(i)=a(i)-b(i)*h1-c(i)*h2;
        h2=h1;
        h1=y(i);
    end
    
x=y;

return