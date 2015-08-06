function [log_delta, l] = reml_delta( GRM, y, covar )
% Optimizes REML equations to find delta

[log_delta, l] = fminbnd(@(x) fast_neg_log_lik(x, covar, y, GRM), -5, 5);

end

