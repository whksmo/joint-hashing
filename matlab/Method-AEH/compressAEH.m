function [B, U] = compressAEH(X, AEHparam)

% Input:
%          X: n*d n is the number of samples, d is the dimension of feature
%          AEHparam: 
%                           AEHparam.nbits---encoding length
%                           AEHparam.w---hashing function
% Output:
%          B: compacted binary code
%          U: binary code

% tic
U = X*AEHparam.w;

B = zeros(size(U));
B(U>=0) = 1;
B(U<0) = 0;
