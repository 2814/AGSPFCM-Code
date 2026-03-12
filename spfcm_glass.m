clear all;
 close all

 %{
 load zoo.txt
     data=zoo(:,1:end-1);
     C=zoo(:,end);
     x=data;
 %}

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

% compute the distances
%{

for j = 1 : cluster_n,
    xv = x - X1*v(j,:);
    d(:,j) = sum((xv*eye(nx).*xv),2);
  end;
  distout=sqrt(d);
  obj_fcn(i) = sum(sum(mf.*d'));

%}

% compute the distances


A=1.4;
b=10;

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
  U=U1';
end



data=x;
a=1;b=1;eta= 2;
t1=rand(cluster_n,mx);
 alpha=0.6;
 
for k=1:max_iter
t=t1.^eta;mf=U.^expo;
 mf1=a*mf+b*t;   sumf1= sum(mf1');
 v1 = (mf1*data)./(sumf1'*ones(1,nx));   
for j = 1 : cluster_n,
    xv = data - X1*v1(j,:);
    dx(:,j) = sum((xv*eye(nx).*xv),2);
end; 
 gamma=sum(mf.*dx',2)./sum(mf,2);
 %const=repmat(gamma',mx,1);
 temp=(1-t1).^eta;
 temp1=gamma.*sum(temp,2);
 obj_fcn1(k)=sum(sum(mf1'.*dx))+sum(temp1);


 y = obj_fcn1;
  x = 1:100;
  plot(x,y,'r','LineWidth',8);
  xlabel('iterations');
  ylabel('obj\_fcn');
  



 if display, 
		fprintf('Iteration count = %d, obj. fcn = %f\n', k, obj_fcn1(k));
    end
  if k > 1,
		if abs(obj_fcn1(k) - obj_fcn1(k-1)) < min_impro, break; end,
    end
  
 temp2 = (b*(dx+1e-10)./repmat(gamma',mx,1)).^(1/(eta-1));
 t2=1./(1+temp2);
 %t1=t2';
 %supressed PCM
  [a1,y]=max(t2,[],2);

  iden=ones(mx,cluster_n);
  for i=1 : mx
    iden(i,y(i))=0;
    
  end
  iden_2=~iden;
  %alpha=0.8;
  alpha_mat=iden .* alpha;
  alpha_1 = t2 .* alpha_mat
  U_new=iden_2 .* t2;
  U_new=U_new + alpha_1;

 t1=U_new';
 
 %% Supressed fcm
 
 dx = (dx).^(-1/(expo-1));
 U2 = (dx ./ (sum(dx,2)*ones(1,cluster_n)));  
 %U=U2';
 
 [a2,y2]=max(U2,[],2);

  iden=ones(mx,cluster_n);
  for i=1 : mx
    iden(i,y2(i))=0;
    
  end
   iden_2=~iden
   %alpha=1;
   alpha_mat=iden .* alpha;
   alpha_1 = U2 .* alpha_mat
   U_new=iden_2 .* U2
   U_new=U_new + alpha_1
  req_sum=sum(U_new .* iden,2);

 for i=1 : mx
    U_new(i,y(i))=1-req_sum(i);
 end
  U=U_new';
end


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





