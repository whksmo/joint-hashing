function LSHparam = trainLSH(LSHparam)

% Input:
%          LSHparam
%              LSHparam.nbits---number of bits (nbits do not need to be a multiple of 8)
% Output:
%             LSHparam:
%                 LSHparam.w---random projection

dim = LSHparam.dim;
nbits = 1*LSHparam.nbits;

W = randn(dim, nbits);
% figure,imshow(W)
LSHparam.w = W;