function yvec1=impf(a,b,shock,nstep,neq);

%*******************************************************
%******** Proc to compute impulse response from system: 
%*******************************************************
y=b*shock;                 % initial value 
yvec=zeros(nstep,length(a));
yvec(1,:)=y';              % store initial value 
i=2;
while i<=nstep;         % loop through periods 
 y=a*y;
 yvec(i,:)=y';
 i=i+1;
end;
yvec1= yvec(:,1:neq);

