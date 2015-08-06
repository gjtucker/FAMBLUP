% y: N x 1 phenotype vector
% GRM: N x N relatedness matrix
% covars: N x C covariate matrix
% X_test: N x Mtest matrix of SNPs to test, coded in EIGENSTRAT format:
%         allele values = 0, 1, 2; missing = 9
%         (see geno2mat.m to convert an EIGENSTRAT .geno file into Matlab format)

% NOTE: Leave-one-chromosome-out testing should be performed as follows:
%       for each chrom
%           compute GRM using all other chromosomes
%           run assoc2VC with X_test = SNPs on left-out chromosome
%       end
% (If X_test for a full chromosome requires too much memory, split it into batches.)

function wald_stats = assoc2VC(y, GRM, covars, X_test)

% compute top 5 PCs of GRM
[U, ~] = eig(GRM); U = U(:, 1:5); 

fixed_effects = orth([ones(size(y)), U, covars]); % Squeeze out the linear dependence

test_GRM = make_GRM_int8(X_test);

[t, IBD] = optimize_t(GRM, test_GRM, U);
fprintf('Threshold = %g\n', t);

test_GRM = [];

%%% MLM_IBD.m (input/output args relabeled):
%%% function [ wald_stats, extra ] = run_GRM(%m%, X_test, y, GRM, IBD, fixed_effects)

            [tau, log_delta_min, res_GRM] = reml_find_tau_delta_grm( ...
                GRM, y, fixed_effects, 0, IBD);
            [~, wald_stats] = mlm_grm(res_GRM, ...
                X_test, y, fixed_effects, false, log_delta_min);
%%% end

end
