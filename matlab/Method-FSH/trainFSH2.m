function FSHparam = trainFSH2(FSHparam)

% Input:
%          LSHparam
%              LSHparam.nbits---number of bits (nbits do not need to be a multiple of 8)
% Output:
%             LSHparam:
%                 LSHparam.w---random projection


dim = FSHparam.dim;
nbits = FSHparam.nbits;
output_dim=20*nbits;
% W = (0.5+0.5*randn(dim, output_dim));
W = rand(dim,output_dim);
% W = (rand(dim, output_dim));
% threshold=0.1+0.005*randn(1,output_dim);
% W=(W<=0.1*dim/output_dim);
% W=(W<=threshold);
% W=(W<=0.1);

% for i=1:size(W,2)
%     W(:,i)=W(:,i)<0.1;%+0.01*(rand()-0.5);
% end
temp=sort(W,1,'descend');
temp1=temp(ceil(0.1*dim),:);
mask1=(W>=temp1);
W(mask1)=1;
W(~mask1)=0;
% save ./myData/WFLSH.mat W
% mean(W)
% std(W)
% sum(W,1)
% figure,imshow(W)
FSHparam.w =W; 
