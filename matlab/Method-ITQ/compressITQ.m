function [B, U] = compressITQ(X, ITQparam)

% Input:
%          X: n*d data matrix, n is number of images, d is dimension
%          ITQparam:
%                           ITQparam.pcaW---PCA of all the database
%                           ITQparam.nbits---encoding length
%                           ITQparam.r---ITQ rotation projection
% output:
%            B: compacted binary code
%            U: binary code

pc = ITQparam.pcaW;
% pc = ITQparam.w;
V = X*pc;

% rotate the data
U = V*ITQparam.r;

B = compactbit(U>0);
B = zeros(size(U));
B(U>=0) = 1;
B(U<0) = -1;

U = (U>0);



