function [B, U] = compressPCAH(X, PCAHparam)
% Input:
%          X: n*d n is the number of samples, d is the dimension of feature
%          PCAHparam:
%                              PCAHparam.nbits---encoding length
%                              PCAHparam.pcaW---hashing function
% Output:
%          B: compacted binary code
%          U: binary code

 
U = X*PCAHparam.pcaW;
B = compactbit(U>0);
% B = zeros(size(U));
% B(U>=0) = 1;
% B(U<0) = -1;

U = (U>0);

