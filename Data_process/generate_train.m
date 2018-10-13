clear;
close all;
folder = 'DIV2K/HR';
savepath = 'data_x2.h5';

%% scale factors
scale = 2;

size_label = 64;
size_input = size_label/scale;
stride = 100;

%% downsizing
downsizes = [1,0.7,0.5];

data = zeros(size_input, size_input, 1, 1);
label = zeros(size_label, size_label, 1, 1);

count = 0;
margain = 0;

%% generate data
filepaths = [];
filepaths = [filepaths; dir(fullfile(folder, '*.jpg'))];
filepaths = [filepaths; dir(fullfile(folder, '*.bmp'))];
filepaths = [filepaths; dir(fullfile(folder, '*.png'))];

length(filepaths)

chunksz = 64;
created_flag = false;
totalct = 0;
batchno = 1;

for i = 1 : length(filepaths)
    for flip = 1: 3
        for degree = 1 : 4
            for downsize = 1 : length(downsizes)
                image = imread(fullfile(folder,filepaths(i).name));
                if flip == 1
                    image = flipdim(image ,1);
                end
                if flip == 2
                    image = flipdim(image ,2);
                end
                
                image = imrotate(image, 90 * (degree - 1));
                image = imresize(image,downsizes(downsize),'bicubic');

                if size(image,3)==3
                    image = rgb2ycbcr(image);
                    image = im2double(image(:, :, 1));

                    im_label = modcrop(image, scale);
                    [hei,wid] = size(im_label);

                    filepaths(i).name
                    for x = 1 + margain : stride : hei-size_label+1 - margain
                        for y = 1 + margain :stride : wid-size_label+1 - margain
                            subim_label = im_label(x : x+size_label-1, y : y+size_label-1);
                            subim_input = imresize(subim_label, 1/scale, 'bicubic');

                            count=count+1;
                            data(:, :, 1, count) = subim_input;
                            label(:, :, 1, count)= subim_label;
                            
                            if mod(count,chunksz) == 0
                                last_read=(batchno-1)*chunksz;
                                startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
                                curr_dat_sz = store2hdf5(savepath, data, label, ~created_flag, startloc, chunksz);
                                created_flag = true;
                                totalct = curr_dat_sz(end);
                                count = 0;
                            end
                        end
                    end
                 end
             end
        end
    end
end

%% writing to HDF5
h5disp(savepath);
