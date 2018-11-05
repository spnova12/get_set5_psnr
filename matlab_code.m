clear; clc;

img_list = dir('.\*.png');

psnr_val = 0;
for index = 1:length(img_list)
    path = [img_list(index).folder, '\', img_list(index).name];
    img = im2double(imread(path));
    img = rgb2ycbcr(img);
    [width, height, ~] = size(img);
    
    % resize and rescale
    img_edited = imresize(img, 0.25, 'bicubic');
    img_edited = imresize(img_edited, 4, 'bicubic');
    
    % remove bdr
    img = img(5:width-4, 5:height-4, 1);
    img_edited = img_edited(5:width-4, 5:height-4, 1);
    
    % cal psnr
    psnr_val = psnr_val + psnr(img_edited, img);
end

disp(psnr_val/length(img_list));