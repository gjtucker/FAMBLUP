function [tau, log_delta_min, res_GRM, log_deltas] = reml_find_tau_delta_grm(WW, y, ...
    covar, IBD_threshold, IBD)
% Takes in the GRM, y, and covariates.  Finds the best tau and delta
% according to REML

if nargin < 4
    IBD_threshold = 0.05;
end

if nargin < 5
    % Create the IBD proxy matrix
    [U, S] = svd(WW);
    S = diag(S);
    S(1:5) = 0;
    IBD = U*diag(S)*U';
    IBD(IBD < IBD_threshold) = 0;
end
   
model = @(x) neg_log_lik_tau(x, WW, IBD, covar, y);

opts = optimset('TolX', 0.01);
tau = fminbnd(model, 0, 1, opts);
[~, log_delta_min] = model(tau);

log_deltas = zeros(2, 1);
[~, log_deltas(1)] = model(0);
[~, log_deltas(2)] = model(1);

res_GRM = tau*WW + (1 - tau)*IBD;

end

function [l, log_delta] = neg_log_lik_tau(tau, WW, IBD, covar, y)
    fprintf('Working on tau = %g\n', tau);

    % Blend IBD and IBS matrices
    GRM = tau*WW + (1 - tau)*IBD;
   
    try
        [log_delta, l] = reml_delta(GRM, y, covar);
    catch err
        fprintf('Error in inner optimization, trying alternative method\n');
        
        % SVD transform problem
        [U, S] = svd(GRM);
        S = diag(S);

        Uy = U'*y;
        Ucovar = U'*covar;

        % Optimize delta
        model = @(x) neg_log_lik(x, Ucovar, Uy, S);
        [log_delta, l] = fminbnd(model, -5, 5);
    end
end
