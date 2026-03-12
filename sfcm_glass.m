% FCM Clustering
clear all
 close all
 
 
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
 
 
%{ 
    load abalone1.txt
    data=abalone1(:,2:end);
    C=abalone1(:,1);
 

    
x=data;

[mx,nx]=size(x);
cluster_n=3;
options = [2;	% exponent for the partition matrix U
		100;	% max. number of iteration
		1e-5;	% min. amount of improvement
		1];	% info display during iteration
 %}
    
expo = options(1);		% Exponent for U
max_iter = options(2);		% Max. iteration
min_impro = options(3);		% Min. improvement
display = options(4);	% Display info or not

% Fuzzy c-means clustering
U=rand(cluster_n,mx);
col_sum=sum(U);
U=U./col_sum(ones(cluster_n,1),:);
obj_fcn=zeros(max_iter,1);
obj_fcn1=zeros(max_iter,1);

% Fuzzy c-means clustering
X1=ones(mx,1);
for i= 1 : max_iter
mf = U.^expo; sumf = sum(mf');
v = (mf*x)./(sumf'*ones(1,nx));

%{
% compute the distances

for p = 1 : cluster_n,
    xv = x - X1*v(p,:);
    d(:,p) = sum((xv*eye(nx).*xv),2);
  end;
  distout=sqrt(d);
  obj_fcn(i) = sum(sum(mf.*d'));

%}
% compute the distances

A=2.4;
b=5.5;

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
  U1 = (d ./ (sum(d,2)*ones(1,cluster_n)));
  [a,y]=max(U1,[],2);

  iden=ones(mx,cluster_n);
  for i=1 : mx
    iden(i,y(i))=0;
    
  end
  iden_2=~iden;
  alpha=0.33
  
  alpha_mat=iden .* alpha;
  alpha_1 = U1 .* alpha_mat
  U_new=iden_2 .* U1;
  U_new=U_new + alpha_1;
  req_sum=sum(U_new .* iden,2);

 for i=1 : mx
    U_new(i,y(i))=1-req_sum(i);
 end
  U=U_new';
end

  y = obj_fcn;
  x = 1:100;
  plot(x,y,'r','LineWidth',8);
  xlabel('iterations');
  ylabel('obj\_fcn');

%Assignment for classification
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
S=mean(s)
u=d2;v=C;
[sse] = clus_sse(d2,x,v);

Randindex=adjrand(u,v);
Norm=nmi1(u,v')
table = [A  S Randindex Norm sum(sse) accuracy];
	colnam = {'Total Misclassification','Silhouette','Rand Index','NMI','SSE', 'Accuracy'};
	h = figure('Name','Metrics','NumberTitle','off','Position',[200 200 400 150]);
	rownam = {'FCM'};
	uitable('Parent',h,'Data',table,'ColumnName',colnam,'RowName',rownam,'Position',[20 20 360 130]);
