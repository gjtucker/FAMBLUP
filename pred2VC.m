% y: N x 1 phenotype vector (including individuals to predict; values will be ignored)
% GRM: N x N relatedness matrix
% covars: N x C covariate matrix
% I_train: N x 1 boolean vector indicating training indivs
%          predictions will be made for I_test = ~I_train

function y_hat = pred2VC(y, GRM, covars, I_train)
% adapted from Risk_Prediction/risk_prediction_rr.m

% compute top 5 PCs of GRM
[U, ~] = eig(GRM); U = U(:, 1:5);

fixed_effects = orth([ones(size(y)), U, covars]); % Squeeze out the linear dependence

n = size(y, 1);

P = eye(n) - U*U'; % projection matrix
GRM = P*GRM*P;
P = []; % clear memory
IBD = GRM;
t = 0.05;
IBD(IBD < t) = 0;

%%% from MLM_IBD.m: (input/output args relabeled):
%%% function [ y_hat, extra ] = predict_untyped(%m%, y, GRM, IBD, fixed_effects, I_train)

            [tau, log_delta_min] = reml_find_tau_delta_grm( ...
                    GRM(I_train, I_train), y(I_train, :), ...
                    fixed_effects(I_train, :), 0, IBD(I_train, I_train));
            fprintf('tau = %f\n', tau);
            GRM = tau*GRM + (1 - tau)*IBD;

            y_hat = predict(GRM, y, fixed_effects, log_delta_min, I_train);
%%% end

end
