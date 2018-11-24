function AEHparam = trainAEH(AEHparam)

% Input:
%          LSHparam
%              LSHparam.nbits---number of bits (nbits do not need to be a multiple of 8)
% Output:
%             LSHparam:
%                 LSHparam.w---random projection

autoenc = AEHparam.aec;

W=autoenc.EncoderWeights';

% histogram(a)
% figure,imshow(W)
AEHparam.w = W;
