
clear;

filelistdir = './data/modellist_modelnet10';
image_dir = './data/imagedata';  %each image of render view
feature_dir = '/data/featuredata'; % the confidence of the trained image 
h5path = './modelnet10_144_train_conf.h5';

labels = {'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet'};
rows = 12;
cols = 12;
featurelen = 10;
for l = 1:size(labels,2) 
    data_disk=zeros(featurelen, cols, rows, 2000); %bhwc
    class = labels{l}
    listfilename = strcat(filelistdir,'/', class , '_train.txt');
    listfile = fopen(listfilename, 'r')
    index = 0;
    while ~feof(listfile)
        index = index + 1;
        modelname = fgetl(listfile);
        for r = 1:rows
            el = 210 - r*30;
            for c = 1:cols
                az = (c-1)*30;
                featurepath = sprintf('%s/%s/train/%s_%03d_%03d.conf', feature_dir, class, modelname, el, az);
                disp(featurepath);
                imnew = load(featurepath);
                imnew = imnew';
                data_disk(:, c, r, index) = imnew;
            end
        end
    end
    fclose(listfile);
    data_disk_fact = data_disk(:,:,:,1:index);
    
    dbname = ['/data/', num2str(l)];
  
    h5create(h5path, dbname, size(data_disk_fact));
    h5write(h5path, dbname, data_disk_fact);
    h5disp(h5path);
end
    
              
disp('done')



