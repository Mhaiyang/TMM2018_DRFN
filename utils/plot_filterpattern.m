function plot_filterpattern(weights, filters, channel, patchsize, framewidth)
mini = min(min(min(weights)));
maxi = max(max(max(weights)));
if channel > 1 || nargin < 5
    framewidth = channel;
end
frame = 255 + zeros(filters*channel/framewidth * patchsize + ...
    (filters * channel/framewidth+1) * 1, framewidth * patchsize + (framewidth + 1) * 1);
for i = 1 : filters * channel/framewidth
    for j = 1 : framewidth
        if channel == 1
           para_tmp = weights(1,:,(i-1)*framewidth+j);
        else
           para_tmp = weights(j,:,i);
        end
        para = reshape(para_tmp,patchsize,patchsize);
        para = (para-mini)/abs(maxi-mini);
        frame((i-1)*(patchsize+1)+2:i*(patchsize+1),...
            (j-1)*(patchsize+1)+2:j*(patchsize+1)) = para';
    end    
end
figure;
imshow(frame);

