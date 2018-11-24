function nn_info=visualize_result(show,query_trueID,retrieval_index,param,img_name,from,position,label)
load ./data/bone/sel.mat
num_visual=param.numRetrieval;
imgData=zeros(num_visual+1,64*64);
prefix = './data/bone';

query_trueID=sel_test(query_trueID);
retrieval_index=sel_train(retrieval_index);

nn_info(1).imname=img_name(query_trueID).from;
if nn_info(1).imname(end-4)=='p'
    nn_info(1).pathname='./HS_SICK_AD/';
else
    nn_info(1).pathname='./HS_NORM_AD/';
end
nn_info(1).from=from(query_trueID).name;
switch position(query_trueID)
    case 0
        nn_info(1).position='Vertebrate';
    case 1
        nn_info(1).position='Left Rib';
    case 2
        nn_info(1).position='Right Rib';
    case 3
        nn_info(1).position='Scapula';
    case 4
        nn_info(1).position='Kidney';  
end
nn_info(1).label=label(query_trueID);
nn_info(1).xCenter=from(query_trueID).xCenter;
nn_info(1).yCenter=from(query_trueID).yCenter;

imnames=[prefix nn_info(1).pathname(2:end) nn_info(1).imname];
img=imread(imnames);
imgData(1,:)=reshape(img,1,[]);
for jj=2:num_visual+1
    nn_info(jj).imname=img_name(retrieval_index(jj-1)).from;
    nn_info(jj).from=from(retrieval_index(jj-1)).name;    
    if nn_info(jj).imname(end-4)=='p'
        nn_info(jj).pathname='./HS_SICK_AD/';
    else
        nn_info(jj).pathname='./HS_NORM_AD/';
    end
    switch position(retrieval_index(jj-1))
        case 0
            nn_info(jj).position='Vertebrate';
        case 1
            nn_info(jj).position='Left Rib';
        case 2
            nn_info(jj).position='Right Rib';
        case 3
            nn_info(jj).position='Scapula';
        case 4
            nn_info(jj).position='Kidney';  
    end
    nn_info(jj).label=label(retrieval_index(jj-1));
    nn_info(jj).xCenter=from(retrieval_index(jj-1)).xCenter;
    nn_info(jj).yCenter=from(retrieval_index(jj-1)).yCenter;
    
    imnames=[prefix nn_info(jj).pathname(2:end) nn_info(jj).imname];
    img=imread(imnames);
    imgData(jj,:)=reshape(img,1,[]);
end
if show
    display_network(imgData(1:end,:),false,true);
end