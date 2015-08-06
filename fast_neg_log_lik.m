function [l, beta, y_resid, V_chol, sigma_g_sq] = fast_neg_log_lik(log_delta, X, y, V)
% Compute restricted maximum likelihood for delta = \sigma_e^2/\sigma_g^2

% S is a column vector
n = length(y);
d = size(X, 2);

delta = exp(log_delta);

for i = 1:n
    V(i, i) = V(i, i) + delta;
end

[V_chol, p] = chol(V);

V_inv = @(A) V_chol \ ((V_chol') \ A);

XVX = X'*(V_inv(X));
[XVX_chol, p] = chol(XVX);
XVX_inv = inv(XVX_chol);
XVX_inv = XVX_inv*XVX_inv';

beta = XVX_inv*(X'*(V_inv(y)));

y_resid = y - X*beta;
sigma_g_sq = 1/(n-d) * (y_resid'*(V_inv(y_resid)));


l = 2*sum(log(diag(V_chol))) + 2*sum(log(diag(XVX_chol))) + (n-d)* ...
    log(sigma_g_sq);

% cast to double because input may be single and fminunc needs doubles
l = double(l);

end