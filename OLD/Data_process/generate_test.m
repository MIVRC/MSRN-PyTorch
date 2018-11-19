clear;close all;
%% settings
folder = 'Rebuild/input/Set5';
scale = 2;

%% generate data
filepaths = [];
filepaths = [filepaths; dir(fullfile(folder, '*.jpg'))];
filepaths = [filepaths; dir(fullfile(folder, '*.bmp'))];
filepaths = [filepaths; dir(fullfile(folder, '*.png'))];

for i = 1 : length(filepaths)        
    im_gt = imread(fullfile(folder,filepaths(i).name));
    im_gt = modcrop(im_gt, scale);
    im_gt = double(im_gt);
    im_gt_ycbcr = rgb2ycbcr(im_gt / 255.0);
    im_gt_y = im_gt_ycbcr(:,:,1) * 255.0;
    
    im_l_ycbcr = imresize(im_gt_ycbcr, 1/scale, 'bicubic');
    im_l_y = im_l_ycbcr(:,:,1) * 255.0;
 
    im_b_ycbcr = imresize(im_l_y, scale, 'bicubic');
    im_b_y = im_b_ycbcr(:,:,1) * 255.0;
    
    filename = sprintf('PSNR_test/2/Set5_x2/%s.mat',filepaths(i).name);
    save(filename, 'im_l_y', 'im_b_y', 'im_gt_y', '-v6');
end
