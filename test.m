% =========================================================================
%  test code using matcaffe
% =========================================================================
%% settings
clear;
addpath ../../matlab/;
addpath utils/;
folder = 'Test/Set5/GT/';
model = 'deploy.prototxt';
weights = 'Caffemodel/DRFN_x4.caffemodel';
batch = 1;
scale = 4;
padding = 8;

%% use gpu mode
caffe.reset_all(); 
caffe.set_mode_gpu();
caffe.set_device(0);

%% read data
n = dir(fullfile(folder,'*.bmp'));

psnr_bic = zeros(length(n),1);
psnr_sr = zeros(length(n),1);
ssim_bic = zeros(length(n),1);
ssim_sr = zeros(length(n),1);
IFC_bic = zeros(length(n),1);
IFC_DRF = zeros(length(n),1);

tic;

net = caffe.Net(model,weights,'test');

for i = 1 : length(n)
    disp(['Process image' num2str(i)]);
    image = imread(fullfile(folder,n(i).name));
    if size(image,3) == 3
        image = rgb2ycbcr(image); 
    end;
    image_luminance = image(:,:,1); % lumination
    image = im2double(image_luminance);
    
    image_label = modcrop(image,scale);
    image_small = imresize(image_label,1/scale,'bicubic');
    
    [height, width, channel] = size(image_small);
    
    net.blobs('data').reshape([height width channel batch]); % reshape blob 'data'
    net.reshape();
    net.blobs('data').set_data(image_small);
    net.forward_prefilled();
    output = net.blobs('hr').get_data();
    
    im_h = shave(double(output * 255) , [padding/2, padding/2]);
    im_gt = shave(image_label * 255 , [padding/2, padding/2]);
    im_b = imresize(image_small,scale,'bicubic');
    im_b = shave(im_b *255 , [padding/2, padding/2]);
    
    IFC_bic(i) = ifcvec(im_gt, im_b);
    if( ~isreal(IFC_bic(i)) )
        IFC_bic(i) = 0;
    end
    IFC_DRF(i) = ifcvec(im_gt, im_h);
    if( ~isreal(IFC_DRF(i)) )
        IFC_DRF(i) = 0;
    end    
    
    im_h = uint8(im_h);
    im_gt = uint8(im_gt);
    im_b = uint8(im_b);
    
    psnr_bic(i) = psnr(im_gt,im_b);
    psnr_sr(i) = psnr(im_gt,im_h);
    
    ssim_bic(i) = ssim(im_gt,im_b);
    ssim_sr(i) = ssim(im_gt,im_h);
       
    cd Test/Set5/results/
    imwrite(im_b,[n(i).name(1:end-4),'_bic.png']);
    imwrite(im_h,[n(i).name(1:end-4),'_sr.png']);
    cd ../../../
    
    clear input;
end 

toc;

%% output
fprintf('Mean PSNR for  Bic:  %.2f dB    SSIM:%.4f   IFC: %f \n', mean(psnr_bic),mean(ssim_bic),mean(IFC_bic));
fprintf('Mean PSNR for  Ours: %.2f dB    SSIM:%.4f   IFC: %f \n', mean(psnr_sr),mean(ssim_sr),mean(IFC_DRF));
