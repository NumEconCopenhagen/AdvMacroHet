function uu = hp2(dat); 
tic
%@ USING A KALMAN FILTER TECHNIQUE TO DERIVE THE H-P FILTER 
%  THE INPUT IS "DAT"==MATRIX OF SERIES THAT ARE DETRENDED 
%                     - REMEMBER TO TAKE LOGS 
%  THE OUTPUT IS "UU"==MATRIX H-P DEVIATIONS FROM TREND   @ 

l=1600; %  SMOOTHING PARAMETER, IT CAN BE CHANGED, 1600 for quarterly

u1 = dat;
[t,ff]=size(dat);
v11 = 1; v22 = 1; v12 = 0; 
i=3; v=zeros(t,3); tt =zeros(t,ff); d = zeros(t,ff); 

for i=3:t %do until i>t; 
 x=v11;  z=v12; 
 v11=(1/l)+4*(x-z)+v22; v12=2*x-z; v22=x; de=v11*v22-v12*v12; 
 v(i,1) = v22/de; v(i,2) = -v12/de; v(i,3) = v11/de; 
 x = v11+1; z=v11; 
 v11 = v11-(v11*v11)/x; v22 = v22-(v12*v12)/x; v12 = v12-(z*v12)/x; 
 %i=i+1; 
end; 

u=u1; m1=u(2,:); m2 = u(1,:); 

i=3; 
for i=3:t; %do until i>t; 
 x = m1; m1=2*m1-m2; m2=x; 
 tt(i-1,:) = v(i,1)*m1+v(i,2)*m2; 
 d(i-1,:) = v(i,2)*m1+v(i,3)*m2; 
 de = v(i,1)*v(i,3)-v(i,2)*v(i,2); 
 v11 = v(i,3)/de; v12 = -v(i,2)/de; 
 z = (u(i,:)-m1)/(v11+1); 
 m1=m1+v11*z; m2=m2+v12*z; 
 %i=i+1; 
end; 

tt(t,:) = m1; tt(t-1,:) = m2; 
m1 = u(t-1,:); m2 = u(t,:); 

i=t-2; 
for i=t-2:(-1):1; %do until i<1; 
 i1=i+1; ib=t-i+1; 
 x=m1; 
 m1=2*m1-m2; m2=x; 
  if i>2; 
   e1 = v(ib,3)*m2+v(ib,2)*m1+tt(i,:); e2 = v(ib,2)*m2+v(ib,1)*m1+d(i,:); 
   b11 = v(ib,3)+v(i1,1); b12 = v(ib,2)+v(i1,2); b22 = v(ib,1)+v(i1,3); 
   de = b11*b22-b12*b12; 
   tt(i,:) = (-b12*e1+b11*e2)/de; 
  end; 
 de = v(ib,1)*v(ib,3)-v(ib,2)*v(ib,2); 
 v11 = v(ib,3)/de; v12 = -v(ib,2)/de; 
 z = (u(i,:)-m1)/(v11+1); 
 m1 = m1 +v11*z; m2 = m2 +v12*z; 
 %i=i-1; 
end; 

tt(1,:) = m1; 
tt(2,:) = m2; 
i=1; 
for i=1:t;%do until i>t; 
 d(i,:) = u(i,:)-tt(i,:); 
 %i=i+1; 
end; 
hu=tt; 
uu = d-(ones(t,ff)*mean(d)')*ones(1,ff); 
%clear u1;
%retp(uu); 
%endp; 
disp(toc);
