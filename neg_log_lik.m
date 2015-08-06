function [l, beta, var_beta, sigma_g_sq] = neg_log_lik(log_delta, UX, Uy, S, use_reml)
% Compute restricted maximum likelihood for delta = \sigma_e^2/\sigma_g^2
% In some rare cases, we may want to compute the actual likelihood, so we
% provide that functionality as an option.

if nargin < 5
    use_reml = true;
end

% S is a column vector
n = length(Uy);
d = size(UX, 2);

% bound log_delta from going crazy
if log_delta > 10
    log_delta = 10;
elseif log_delta < -10
    log_delta = -10;
end

delta = exp(log_delta);
V_inv = 1./(S + delta);

% XVX must be positive definite
XV_sqrt = bsxfun(@times, sqrt(V_inv), UX);
XVX = XV_sqrt'*XV_sqrt; %UX'*bsxfun(@times, V_inv, UX);
[XVX_chol, p] = chol(XVX);

if p > 0
    disp(delta)
    disp(XVX)
    disp(S)
end

XVX_inv = inv(XVX_chol);
XVX_inv = XVX_inv*XVX_inv';

beta = XVX_inv*(UX'*(V_inv.*Uy));


if use_reml
    sigma_g_sq = 1/(n-d) * sum((Uy - UX*beta).^2.*V_inv);

    l = sum(log(S + delta)) + 2*sum(log(diag(XVX_chol))) + (n-d)* ...
        log(sigma_g_sq);
else
    sigma_g_sq = 1/n * sum((Uy - UX*beta).^2.*V_inv);

    l = (sum(log(S + delta)) + n*log(sigma_g_sq) + n + n*log(2*pi))/2;
end
    
var_beta = XVX_inv;

% cast to double because input may be single and fminunc needs doubles
l = double(l);

end