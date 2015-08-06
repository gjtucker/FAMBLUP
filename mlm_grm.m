function [ lambda, wald_stats, p_val, numerator, denominator, delta_hist ] = mlm_grm(GRM, X_test, y, covar, ...
    refit_delta, log_delta_min)
%{
FaST-LMM without proximal contamination.
 
IMPORTANT: W and X_test assumed to be disjoint! (like in LOCO)

Optimize delta coarsely for the full GRM and no test SNP

For each SNP:
    reoptimize delta with that SNP as a fixed effect + other covars
    get beta_hat, var_beta_hat and sigmga_g
    calc stat and store it

%}

if nargin < 5
    refit_delta = true;
end
    
if nargin < 6
    fixed_delta = false;
else
    fixed_delta = true;
end

if refit_delta && fixed_delta
    fprintf('Possible error refit delta and delta fixed are set.  Defaulting to refitting delta.');
    fixed_delta = false;
end

% SVD transform problem
[U, S] = svd(GRM);
S = diag(S); 

Uy = U'*y;
Ucovar = U'*covar;

if ~fixed_delta
    % Optimize delta
    model = @(x) neg_log_lik(x, Ucovar, Uy, S);
    log_delta_min = fminbnd(model, -5, 5);
end

% I don't believe the following normalization is necessary
W_test = double(X_test);
m_test = size(W_test, 2);
wald_stats = zeros(m_test, 1);
numerator = zeros(m_test, 1);
denominator = zeros(m_test, 1);
delta_hist = zeros(m_test, 1);

UW_test = U'*W_test;
for i = 1:m_test
    % Compute statistic for SNP i
    Q = [UW_test(:, i), Ucovar];
    
    % Refine estimate of delta for this particular SNP
    snp_model = @(x) neg_log_lik(x, Q, Uy, S);
        
    if ~fixed_delta && refit_delta
        try
            snp_log_delta = fminunc(snp_model, log_delta_min, fmin_opts);
        catch err
            disp(log_delta_min)
            disp(Q)
            rethrow(err);
        end
    else
        snp_log_delta = log_delta_min;
    end
    
    [~, beta, var_beta, sigma_g_sq] = snp_model(snp_log_delta);
    
    wald_stats(i) = beta(1)^2/(sigma_g_sq*var_beta(1, 1));
    numerator(i) = beta(1)^2/sigma_g_sq;
    denominator(i) = var_beta(1, 1);
    delta_hist(i) = exp(snp_log_delta);
end

lambda = lambda_GC(wald_stats);
p_val = 1 - chi2cdf(wald_stats, 1);

end