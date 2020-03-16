function [Z,C] = DNRC(X, Y, param)
lambda = param.lambda;
mu = param.mu;
[~,n] = size(X);
m = size(Y,2);
tol = 1e-5;
maxIter = 5;
% mu= 1e-1;
Z = zeros(n,m);
C = zeros(n,m);
delta = zeros(n,m);

XTX = X'*X;
XTY = X'*Y;
iter = 0;

% class_num = length(unique(tr_label));
% tr_sym_mat = zeros(n);
% for ci = 1 : class_num
%     ind_ci = find(tr_label == ci);
%     tr_descr_bar = zeros(size(X));
%     tr_descr_bar(:,ind_ci) = X(:, ind_ci);
%     tr_sym_mat = tr_sym_mat + lambda * (tr_descr_bar' * tr_descr_bar);
% end

temp_X = pinv((1+lambda)*XTX+mu/2*eye(n));
% temp_X = inverse((1+lambda)*XTX+mu/2*eye(n));

while iter<maxIter
    iter = iter + 1;
    
    Zk = Z;
    Ck = C;
    
    % update c
    %     c = (XTX+tr_sym_mat+mu/2*eye(n))\((1+lambda)*XTy+mu/2*z+delta/2);
    C = temp_X*(XTY+mu/2*Z+delta/2);
    
    % update z
    z_temp = C-delta/mu;
    Z = max(0,z_temp);
    
    leq1 = Z-C;
    leq2 = Z-Zk;
    leq3 = C-Ck;
    stopC1 = max(norm(leq1,'fro'),norm(leq2,'fro'));
    stopC = max(stopC1,norm(leq3,'fro'));
    %     disp(stopC)
    
    if stopC<tol || iter>=maxIter
        break;
    else
        delta = delta + mu*leq1;
    end
end