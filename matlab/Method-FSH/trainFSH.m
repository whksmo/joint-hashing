function FSHparam = trainFSH(FSHparam)

% Input:
%          LSHparam
%              LSHparam.nbits---number of bits (nbits do not need to be a multiple of 8)
% Output:
%             LSHparam:
%                 LSHparam.w---random projection


dim = FSHparam.dim;
nbits = FSHparam.nbits;
output_dim=20*nbits;
W = (0.5+0.315*randn(dim, output_dim));

% W=(W<=0.1*dim/output_dim);
% W=(W<=0.1);

% global aec
% W=aec.EncoderWeights';

temp=sort(W,1,'descend');

% W = rand(dim,output_dim);
% global ss
% stds=ss*0.001;
% for i=1:size(W,2)
%     W(:,i)=W(:,i)<0.1+stds*(randn()-0.5);
% %     W(:,i)=W(:,i)<0.1;
% end
rand_smaple=ceil((0.1+0.014*randn(1,output_dim))*dim);%ÿһάҪ�����ĸ���
rand_smaple(rand_smaple<1)=1;
temp1=temp(rand_smaple,1:output_dim);
temp1=diag(temp1)';
mask1=(W>=temp1);
W(mask1)=1;
W(~mask1)=0;
% mean(W)
% std(W)

% save ./myData/WpFLSH.mat W
% a=sum(W,1); 
% std(a)

% histogram(a)
% figure,imshow(W)
FSHparam.w = W;
