function y_hat = predict(GRM, y, covar, log_delta, I_train, snpGRM)
% BLUP prediction
% 
% If I_train not given, then train and predict on all data.

if nargin < 5
    I_train = 1:length(y);
    I_pred = I_train;
else
    I_pred = ~I_train;
end

if nargin < 4
    log_delta = reml_delta(GRM, y, covar);
end

delta = exp(log_delta);

% Compute V
for i = 1:size(GRM, 1)
    GRM(i, i) = GRM(i, i) + delta;
end

% Compute alpha
% 1) Compute V^{-1}*Q
V_inv_Q = GRM(I_train, I_train) \ covar(I_train, :);

% 2) Compute V^{-1}*y_train
V_inv_y = GRM(I_train, I_train) \ y(I_train);

alpha = (covar(I_train, :)'*V_inv_Q) \ (covar(I_train, :)'*V_inv_y);

% Compute y_pred
if nargin == 5
    y_hat = GRM(I_pred, I_train)*(V_inv_y - V_inv_Q*alpha) + covar(I_pred, :)*alpha;
else % predict using betas only
    y_hat = snpGRM(I_pred, I_train)*(V_inv_y - V_inv_Q*alpha) + covar(I_pred, :)*alpha;
end

end