% FCM Clustering
clc;
clear all;
close all;

%{

    load abalone1.txt
    data=abalone1(:,2:end);
    C=abalone1(:,1);
    
    
x=data;
%}

 
 load WINEDAT.TXT
     data=WINEDAT(:,1:end-1);
     C=WINEDAT(:,end);
     x=data;


[mx,nx]=size(x);
cluster_n=3;
options = [2;	% exponent for the partition matrix U
		100;	% max. number of iteration
		1e-5;	% min. amount of improvement
		1];	% info display during iteration 
    
expo = options(1);		% Exponent for U
max_iter = options(2);		% Max. iteration
min_impro = options(3);		% Min. improvement
display = options(4);	% Display info or not

% FCM Clustering
X1=ones(mx,1);
obj_fcn = zeros(max_iter, 1);	% Array for objective function
obj_fcn1 = zeros(max_iter, 1);

% initialize partition matrix

U=rand(cluster_n,mx);
col_sum=sum(U);
U=U./col_sum(ones(cluster_n,1),:);
% compute the cluster prototypes
for i= 1 : max_iter
mf = U.^expo; sumf = sum(mf');
v = (mf*x)./(sumf'*ones(1,nx));

%{
% compute the distances

for j = 1 : cluster_n,
    xv = x - X1*v(j,:);
    d(:,j) = sum((xv*eye(nx).*xv),2);
  end;
  distout=sqrt(d);
  obj_fcn(i) = sum(sum(mf.*d'));

%}

% compute the distances


A=1.4;
b=3.5;

for p = 1 : cluster_n,
    xv = x - X1*v(p,:);
    d(:,p) = sum((xv*eye(nx).*xv),2);
  end;
  distout=sqrt(d);

  t=(-b).*(d'.^2);
  %obj_fcn(i) = sum(sum(mf.*d'));
  %obj_fcn(i) = sum(sum(mf.*(1-A.^(-b.*(d'.^2)))));
  obj_fcn(i) = sum(sum(sqrt(1-A.^t)));


  if display, 
		fprintf('Iteration count = %d, obj. fcn = %f\n', i, obj_fcn(i));
	end
  if i > 1,
		if abs(obj_fcn(i) - obj_fcn(i-1)) < min_impro, break; end,
	end
  
  % Update the partition matrix
  d = (d+1e-10).^(-1/(expo-1));
  %d1=(d).^(-1/expo-1)
  U1 = (d ./ (sum(d,2)*ones(1,cluster_n)));
  U=U1';
end

%Possibilistic c-means clustering

U=U.^expo;
sumf1=sum(U'); 
 % Calculate eta
 for i=1:cluster_n
  dx=x-repmat(v(i,:),mx,1);
  dxsq=sum(dx.^2,2);
  dist=U(i,:)*dxsq;
  eta(i)=dist/sumf1(i);
end

for k1=1:max_iter
U2=U.^expo ; sumf=sum(U2');

% Calculate center
v1 = (U2*x)./(sumf'*ones(1,nx));

% Calculate distance

for k=1:mx
    for i=1:cluster_n
        sumf2=0;
        for j=1:nx
            sumf2=(x(k,j)-v1(i,j))^2+sumf2;
            temp=sumf2;
        end
        d2(i,k)=temp;
    end
end
  
 
% Calculate objective function
 obj_fcn1(k1) = sum(sum(U2.*d2))+eta*sum((1-U2).^2,2);
  if display, 
		fprintf('Iteration count = %d, obj. fcn = %f\n', k1, obj_fcn1(k1));
    end
  if k1 > 1,
		if abs(obj_fcn1(k1) - obj_fcn1(k1-1)) < min_impro, break; end,
    end 
 temp1=d2'./repmat(eta,mx,1);
 U3=1./(1+temp1.^(1/(expo-1)));
 
 %U=U3';
 % Suppressed PCM
 
 [a,y]=max(U3,[],2);

  iden=ones(mx,cluster_n);
  for i=1 : mx
    iden(i,y(i))=0;
    
  end
  iden_2=~iden;
  alpha=0.33
  alpha_mat=iden .* alpha;
  alpha_1 = U3 .* alpha_mat
  U_new=iden_2 .* U3;
  U_new=U_new + alpha_1;
  U=U_new';
end


  y = obj_fcn1;
  x = 1:100;
  plot(x,y,'r','LineWidth',8);
  xlabel('iterations');
  ylabel('obj\_fcn');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %Assignment for classification
 % %  Assignment for classification
[d1,d2]=max(U);
Cc=[];
for i=1:cluster_n
    Ci=C(find(d2==i));
    dum1=hist(Ci,1:cluster_n);
    [dd1,dd2]=max(dum1);
    Cc(i)=dd2;
end
% %%%%%a
for i=1:max(C)
    
    index=find(C==i);
    count(i)=prod(size((index)));   
    err=(Cc(d2(index))~=i);
    eindex=find(err);
    misclass(i)=sum(err);
   
end 
misclass


% ACCURACY AND ERROR
sum1=0;
sume=0;
for i=1:max(C)
    countc(i)=count(i)-misclass(i);
    sum1=sum1+countc(i);
    sume=sume+misclass(i);
end
for i=1:cluster_n
         fprintf('No. of data points in cluster %d= %d\n',i,countc(i));
end
accuracy=(sum1/mx)*100;
fprintf('Accuracy= %f%%\n\n',accuracy);
error=(sume/mx)*100;
fprintf('Error Rate= %f%%\n\n',error);

A=sum(misclass)
s=silhouette(x,d2');
S=sum(s)
u=d2;v=C;
[C1 sse] = clus_sse(d2,x);

Randindex=adjrand(u,v);
Norm=nmi1(u,v')
table = [A  S Randindex Norm sum(sse) accuracy];
	colnam = {'Total Misclassification','Silhouette','Rand Index','NMI','SSE', 'Accuracy'};
	h = figure('Name','Metrics','NumberTitle','off','Position',[200 200 400 150]);
	rownam = {'PCM'};
	uitable('Parent',h,'Data',table,'ColumnName',colnam,'RowName',rownam,'Position',[20 20 360 130]);
