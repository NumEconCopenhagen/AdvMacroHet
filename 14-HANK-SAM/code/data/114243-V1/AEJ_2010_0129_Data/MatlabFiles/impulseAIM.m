%% computing impulse responses
%% maximum lags allowed in system == maxlag = 8 in this program

[mm,nn]=size(b)
%nbeq=10-1  %yo cambie aqui
nbeq=size(b,1)-nsh

%nrrpsi
%lmy
neq=nbeq+nsh		

nvars=nbeq;		
nlag
b;

nt=500+nlag;
e=zeros(nt,nsh);
x=zeros(nt,nbeq);	
e(1+1,shocknum) = 1; %aqui iba un 8+1


if nlag ==  8;
amat1=b(1:nbeq,7*neq+1:7*neq+nbeq);
amat2=b(1:nbeq,6*neq+1:6*neq+nbeq);
amat3=b(1:nbeq,5*neq+1:5*neq+nbeq);
amat4=b(1:nbeq,4*neq+1:4*neq+nbeq);
amat5=b(1:nbeq,3*neq+1:3*neq+nbeq);
amat6=b(1:nbeq,2*neq+1:2*neq+nbeq);
amat7=b(1:nbeq,1*neq+1:1*neq+nbeq);
amat8=b(1:nbeq,0*neq+1:0*neq+nbeq);
end
if nlag ==  7;
amat1=b(1:nbeq,6*neq+1:6*neq+nbeq);
amat2=b(1:nbeq,5*neq+1:5*neq+nbeq);
amat3=b(1:nbeq,4*neq+1:4*neq+nbeq);
amat4=b(1:nbeq,3*neq+1:3*neq+nbeq);
amat5=b(1:nbeq,2*neq+1:2*neq+nbeq);
amat6=b(1:nbeq,1*neq+1:1*neq+nbeq);
amat7=b(1:nbeq,0*neq+1:0*neq+nbeq);
amat8=0*b(1:nbeq,0*neq+1:0*neq+nbeq);
end
if nlag ==  6;
amat1=b(1:nbeq,5*neq+1:5*neq+nbeq);
amat2=b(1:nbeq,4*neq+1:4*neq+nbeq);
amat3=b(1:nbeq,3*neq+1:3*neq+nbeq);
amat4=b(1:nbeq,2*neq+1:2*neq+nbeq);
amat5=b(1:nbeq,1*neq+1:1*neq+nbeq);
amat6=b(1:nbeq,0*neq+1:0*neq+nbeq);
amat7=0*b(1:nbeq,0*neq+1:0*neq+nbeq);
amat8=0*b(1:nbeq,0*neq+1:0*neq+nbeq);
end
if nlag ==  5;
amat1=b(1:nbeq,4*neq+1:4*neq+nbeq);
amat2=b(1:nbeq,3*neq+1:3*neq+nbeq);
amat3=b(1:nbeq,2*neq+1:2*neq+nbeq);
amat4=b(1:nbeq,1*neq+1:1*neq+nbeq);
amat5=b(1:nbeq,0*neq+1:0*neq+nbeq);
amat6=0*b(1:nbeq,0*neq+1:0*neq+nbeq);
amat7=0*b(1:nbeq,0*neq+1:0*neq+nbeq);
amat8=0*b(1:nbeq,0*neq+1:0*neq+nbeq);
end
if nlag ==  4;
amat1=b(1:nbeq,3*neq+1:3*neq+nbeq);
amat2=b(1:nbeq,2*neq+1:2*neq+nbeq);
amat3=b(1:nbeq,1*neq+1:1*neq+nbeq);
amat4=b(1:nbeq,0*neq+1:0*neq+nbeq);
amat5=0*b(1:nbeq,0*neq+1:0*neq+nbeq);
amat6=0*b(1:nbeq,0*neq+1:0*neq+nbeq);
amat7=0*b(1:nbeq,0*neq+1:0*neq+nbeq);
amat8=0*b(1:nbeq,0*neq+1:0*neq+nbeq);
end
if nlag ==  3;
amat1=b(1:nbeq,2*neq+1:2*neq+nbeq);
amat2=b(1:nbeq,1*neq+1:1*neq+nbeq);
amat3=b(1:nbeq,0*neq+1:0*neq+nbeq);
amat4=0*b(1:nbeq,0*neq+1:0*neq+nbeq);
amat5=0*b(1:nbeq,0*neq+1:0*neq+nbeq);
amat6=0*b(1:nbeq,0*neq+1:0*neq+nbeq);
amat7=0*b(1:nbeq,0*neq+1:0*neq+nbeq);
amat8=0*b(1:nbeq,0*neq+1:0*neq+nbeq);
end
if nlag ==  2;
amat1=b(1:nbeq,1*neq+1:1*neq+nbeq);
amat2=b(1:nbeq,0*neq+1:0*neq+nbeq);
amat3=0*b(1:nbeq,0*neq+1:0*neq+nbeq);
amat4=0*b(1:nbeq,0*neq+1:0*neq+nbeq);
amat5=0*b(1:nbeq,0*neq+1:0*neq+nbeq);
amat6=0*b(1:nbeq,0*neq+1:0*neq+nbeq);
amat7=0*b(1:nbeq,0*neq+1:0*neq+nbeq);
amat8=0*b(1:nbeq,0*neq+1:0*neq+nbeq);
end
if nlag ==  1;
amat1=b(1:nbeq,0*neq+1:0*neq+nbeq); 
amat2=0*b(1:nbeq,0*neq+1:0*neq+nbeq);
amat3=0*b(1:nbeq,0*neq+1:0*neq+nbeq);
amat4=0*b(1:nbeq,0*neq+1:0*neq+nbeq);
amat5=0*b(1:nbeq,0*neq+1:0*neq+nbeq);
amat6=0*b(1:nbeq,0*neq+1:0*neq+nbeq);
amat7=0*b(1:nbeq,0*neq+1:0*neq+nbeq);
amat8=0*b(1:nbeq,0*neq+1:0*neq+nbeq);
end
if nlag ==  0;
amat1=0*b(1:nbeq,0*neq+1:0*neq+nbeq);
amat2=0*b(1:nbeq,0*neq+1:0*neq+nbeq);
amat3=0*b(1:nbeq,0*neq+1:0*neq+nbeq);
amat4=0*b(1:nbeq,0*neq+1:0*neq+nbeq);
amat5=0*b(1:nbeq,0*neq+1:0*neq+nbeq);
amat6=0*b(1:nbeq,0*neq+1:0*neq+nbeq);
amat7=0*b(1:nbeq,0*neq+1:0*neq+nbeq);
amat8=0*b(1:nbeq,0*neq+1:0*neq+nbeq);
end

nsh
nbeq
bmat=sinv(1:nbeq,nbeq+1:nbeq+nsh)



     if nlag <1+1;    %aqui va 8+1 en los dos, y sacarle los % y al xc
for i = 1+1:nt,

        x1=x(i-1,:)';	
        %x2=x(i-2,:)';	
        %x3=x(i-3,:)';	
        %x4=x(i-4,:)';	
        %x5=x(i-5,:)';	
        %x6=x(i-6,:)';	
        %x7=x(i-7,:)';	
        %x8=x(i-8,:)';	
        
        ii=size(e);
        e(i,:);
        ee=e(i,:)';
        xc=amat1*x1;
%+amat2*x2+amat3*x3+amat4*x4+amat5*x5+amat6*x6+amat7*x7+amat8*x8;
	ll=size(xc);
        ff=size(bmat);
        
        gg=size(ee);
        ee;
                xc=xc+bmat*ee;
		for k = 1:nbeq,
		    if abs(xc(k,1)) < 0.00000001,
		       xc(k,1) = 0;
		    end;
                   end;

 

        x(i,:)=xc';

end;

diary policy3.txt
matrix=x(1:500,1:4);       % la manera de obtener el impuse response
diary off

impact=x(2,1);
     kehoe=x(214,1);

halflife=kehoe/impact;

rat=1.67/2.99;

%iff=matrix(10,17 )- (1/0.24)*matrix(10,8 )


porcyMC=x(2,10)/x(2,1);
    
else
test1 = input('Error in Computing impulse responses: Too many lags in System (enter 1 to abort)')
end;


   
%x1=x(:,6);
%X=[x1(1:50,:) x1(51:100,:) x1(101:150,:) x1(151:200,:) x1(201:250,:) x1(251:300,:)];




%gama
%zeta

                           





