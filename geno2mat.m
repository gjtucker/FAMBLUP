function geno = geno2mat(in_path,out_path)

fid = fopen(in_path);
C = textscan(fid,'%s',1, 'BufSize', 2^20); % 1x1 cell array containing first line
fclose(fid);
N = length(C{1}{1});
fprintf('num individuals: N = %d\n',N);

fid = fopen(in_path);
geno = int8(fscanf(fid,'%s')); % single string containing all data -> int8
fclose(fid);
M = length(geno)/N;
fprintf('num SNPs: M = %d\n',M);
geno = reshape(geno-int8('0'),N,M);

save(out_path,'geno','-v7.3');

end