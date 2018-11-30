function nn_info = visualize_cifar100(ind, param)   
top_k = param.numRetrieval;
query_id=param.query_id;  

imgsize = 32;
channels = 3;
imgdata=zeros(top_k+1, imgsize * imgsize, channels);
dataset_path = '/data/images/cifar-100/';

tst_file_list = textscan(fopen('./data/cifar100/val.txt'), '%s%d%d', 'delimiter', ' ');
tst_label_list(:,1) = tst_file_list{1,2};
tst_label_list(:,2) = tst_file_list{1,3};
tst_file_list = tst_file_list{1,1};
query_img_name = tst_file_list(query_id);
query_img_name = query_img_name{1};


nn_info(1).imname=query_img_name;
nn_info(1).label=tst_label_list(query_id,:);
imnames=[dataset_path query_img_name];
img=imread(imnames);
imgdata(1,:,:)=reshape(img,1,[], channels);


trn_file_list = textscan(fopen('./data/cifar100/train.txt'), '%s%d%d', 'delimiter', ' ');
trn_label_list(:,1) = trn_file_list{1,2};
trn_label_list(:,2) = trn_file_list{1,3};
trn_file_list = trn_file_list{1,1};
retrieval_index = ind(query_id,1:top_k);
for i=2:top_k+1
    trn_file_name = trn_file_list(retrieval_index(i-1));
    nn_info(i).imname = trn_file_name{1};
    nn_info(i).label=trn_label_list(retrieval_index(i-1),:);
    imnames=[dataset_path nn_info(i).imname];
    img=imread(imnames);
    imgdata(i,:,:)=reshape(img,1,[],channels);
end
if ~isfield(param,'hidePic')
    display_network(imgdata(1:end,:,:),false,false,true);
end

end