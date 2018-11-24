function [B, U] = compressFSH2(X, FSHparam)

% Input:
%          X: n*d n is the number of samples, d is the dimension of feature
%          LSHparam: 
%                           LSHparam.nbits---encoding length
%                           LSHparam.w---hashing function
% Output:
%          B: compacted binary code
%          U: binary code

topN=FSHparam.nbits;
U = X*FSHparam.w;
% U=U';%[2000*sampleNum]
temp=sort(U,2,'descend');
temp=temp(:,topN);
U=(U>=temp);
% U=reshape(U,topN,size(X,1));
% U=floor(U/2);
% U=U';
B = zeros(size(U));
B (U) = 1;
B (~U) = -1;
% sum(U,2)
% save ./myData/UFLSH.mat U 
% a=sum(U,1);
% fig,histogram(a);
% fig,plot(a);
% B = compactbit(U);

