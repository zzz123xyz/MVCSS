function [clusters, obj_value, F_record] = multi_view_fusion(X, c, gamma1, gamma2)

%% =================================
% for implimentation of 
% Wang, Hua, Feiping Nie, and Heng Huang. "Multi-view clustering and feature learning via structured sparsity." 
% Proceedings of the 30th International Conference on Machine Learning (ICML-13). 2013.
% Lance (Liangchen) Liu

%d: dim 
%n: number of sample
%c: number of clusters
%X_a: R^{d,n} 
%W: R^{d,c} 
%% ===================================
 iters = 10;

 k = numel(X);
 Vd = cellfun('size',X,1); % vector d_j, the j-th dim 
 X_a = cell2mat(X');  %X_a means X{i} all concatenate together
 
 [d,n] = size(X_a);
 W = rand(d,c);
 V1n = ones(n,1);
 b = rand(c,1);
 

 [idx, ~, ~, kmeans_dist] = kmeans(X_a', c);
 kmeans_likelihood = 1./kmeans_dist;
 temp1 = repmat(sum(kmeans_likelihood'),[c,1]);
 F0 = kmeans_likelihood./temp1';  %compute the how likely xi belongs to the j-th cluster
 
 b = F0'*V1n/n; %because the data are centered
 
 % initiate 
 for i = 1:c
     b_i = repmat(b(i),[n,1]);
     W(:,i) = inv(X_a*X_a') * X_a*(F0(:,i) - b_i); 
     %W(:,i) = (X_a*X_a') \ X_a*(F0(:,i) - b_i); 
 end    
     
 % loop start
 for t = 1:iters     
     A = X_a'*W + V1n*b';
     [U,S,V] = svd(A);
     I = eye(c);
     M0 = zeros(n-c,c);
     F = U*[I;M0]*V'; 

     b = F'*V1n/n;
     
 % compute the section starting point and ending point of w_i^j;    
     L_21normW = 0;
     for i = 1:size(W,1)
         Vwi(i) = 1/(2*norm(W(i,:)));
         L_21normW = L_21normW + norm(W(i,:));
     end
     D_t = diag(Vwi);
 
     temp1 = [];
     temp2 = 0;
     for i = 1:numel(Vd) 
         temp2 = Vd(i)+temp2;
         temp1(i) = temp2;
     end
     sec_s = [1,temp1(1:end-1)+1]; %section starting point
     sec_e = temp1;
     
     temp1 = 0;
     L_GnormW = 0;
     for i = 1:c
         for j = 1:k
             Vwij = W(sec_s(j):sec_e(j),i);
             Cblkdiag{j} = 1/(2*norm(Vwij))*eye(Vd(j));
             
             temp1 = temp1 + norm(Vwij); %inside loop of G_norm
         end
         L_GnormW = L_GnormW + temp1; %outside loop of G_norm
         
         %      % another way to compute D_t
         %      temp = diag(W*W').^(1/2);
         %      Vwi = 1./(2.*temp);
         
         D_i{i} = blkdiag(Cblkdiag{:}); % block diag matrix Di
         
         b_i = repmat(b(i),[n,1]);
         W(:,i) = inv(X_a*X_a'+gamma1*D_i{i}+gamma2*D_t) * X_a*(F(:,i) - b_i);
     end
     
     F_record{t} = F;
     b_record{t} = b;
     W_record{t} = W;   
     
     obj_value(t) = norm(X_a'*W + V1n*b' - F,'fro')^2 + gamma1 * L_GnormW + gamma2 * L_21normW;
 end
 
 
clusters = kmeans(F_record{end}, c);
 
 
 
 
