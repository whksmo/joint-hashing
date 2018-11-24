function [B, U] = compressFSH(X, FSHparam)

% Input:
%          X: n*d n is the number of samples, d is the dimension of feature
%          LSHparam: 
%                           LSHparam.nbits---encoding length
%                           LSHparam.w---hashing function
% Output:
%          B: compacted binary code
%          U: binary code

topN=FSHparam.nbits;
% tic
U = X*FSHparam.w;
% toc
% U=U';%[2000*sampleNum]
temp=sort(U,2,'descend');
% for i=1:size(U,1)
%     topN_rand=ceil(topN*(1+0.1*(randn()-0.5)));
%     
%     t=temp(i,topN_rand);
%     U(i,1:topN)=1;
% end

temp=temp(:,topN);
U=(U>=temp);
B = zeros(size(U));
% U=(U==1);
B (U) = 1;
B (~U) = -1;
% sum(U,2)
% save ./myData/UpFLSH.mat U 
% a=sum(U,1);
% fig,histogram(a)
% fig,plot(a);
% B = compactbit(U);

