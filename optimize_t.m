function [ t, IBD, GRM_without_U ] = optimize_t( GRM, test_GRM, U )
% Finds the optimal threshold

if nargin < 3
    [U, ~] = svd(GRM + test_GRM);
    U = U(:, 1:5); % take U to be the top 5 PCs
end

n = size(GRM, 1);
P = eye(n) - U*U'; % projection matrix

GRM_without_U = P*GRM*P;
test_GRM = P*test_GRM*P;

score = @(t) sum((test_GRM(GRM_without_U > t) - GRM_without_U(GRM_without_U > t)).^2) + ...
    sum(test_GRM(GRM_without_U <= t).^2);

t = fminbnd(score, 0.02, 0.5);
IBD = GRM_without_U;
IBD(IBD < t) = 0;

end

