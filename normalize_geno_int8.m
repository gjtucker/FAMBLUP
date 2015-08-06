% normalize 0129-matrix
% fill in missing values with column means; then normalize cols to var=1
function [X,orig_stds] = normalize_geno_int8(geno)

N = size(geno,1); M = size(geno,2);
X = single(zeros(N,M));
orig_stds = single(zeros(M,1));

for m = 1:M
    col = single(geno(:,m));
    missing = col==9;
    mu = mean(col(~missing));
    col(missing) = mu;

    % std(.) computes sample standard deviation; std(.,1) computes pop std
    %sigma = std(col,1); % normalize after filling in missing values
    sigma = sqrt(mu*(1 - mu/2));
    
    if sigma < 1e-9 % snp is fixed
        sigma = single(9);
    end
    X(:,m) = (col-mu) / sigma;
    orig_stds(m) = sigma;
end

end
