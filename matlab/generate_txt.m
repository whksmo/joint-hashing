load ./data/bone/label.mat
load ./data/bone/sel.mat
load ./data/bone/from.mat
%%
% label_matrix=position;
label_matrix = label;
test_imlist=cnn_each(sel_test);
train_imlist=cnn_each(sel_train);
train_imlist_cell=struct2cell(train_imlist);
train_imlist_array=struct2array(train_imlist);
%%
fullpath='/data/bone/';

f=fopen('./data/bone/val-disease.txt','wt');
% fls=fopen('/home/xukuan/projects/caffe-SSDH/data/bone/test-file-list.txt','wt');
flb=fopen('./data/bone/test-label-disease.txt','wt');
for i = 1:length(sel_test)
    imgname=test_imlist(i).from;
    if imgname(end-4)=='p'
        pathname='HS_SICK_AD/';
    else
        pathname='HS_NORM_AD/';
    end
    imglabel=label_matrix(sel_test(i));
    fprintf(f,'%s\n',[pathname imgname ' ' num2str(imglabel)]);
%     fprintf(fls,'%s\n',[fullpath pathname imgname]);
    fprintf(flb,'%s\n',[num2str(imglabel)]);
end
fclose('all');
%%
f=fopen('./data/bone/train-disease.txt','wt');
% fls=fopen('/home/xukuan/projects/caffe-SSDH/data/bone/train-file-list.txt','wt');
flb=fopen('./data/bone/train-label-disease.txt','wt');
for i = 1:length(sel_train)
    imgname=train_imlist(i).from;
    if imgname(end-4)=='p'
        pathname='HS_SICK_AD/';
    else
        pathname='HS_NORM_AD/';
    end
    imglabel=label_matrix(sel_train(i));
    fprintf(f,'%s\n',[pathname imgname ' ' num2str(imglabel)]);
%     fprintf(fls,'%s\n',[fullpath pathname imgname]);
    fprintf(flb,'%s\n',[num2str(imglabel)]);
    
end
fclose('all');
