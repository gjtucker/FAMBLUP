function [Theta, Mpoly] = make_GRM_int8(geno)
    N = size(geno,1); M = size(geno,2);
    Theta = single(zeros(N));
    BLOCK_SIZE = 1000;
    Mpoly = 0;
    for mStart = 1:BLOCK_SIZE:M
        mEnd = min(M, mStart+BLOCK_SIZE-1);
        fprintf('adding contribution from %d:%d...\n',mStart,mEnd);
        [X,orig_stds] = normalize_geno_int8(geno(:,mStart:mEnd));
        Theta = Theta + X*X';
        Mpoly = Mpoly + sum(orig_stds ~= single(9));
    end
    Mpoly = single(Mpoly);
    Theta = Theta / Mpoly;
end
