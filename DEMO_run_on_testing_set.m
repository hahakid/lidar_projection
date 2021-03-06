clear; dbstop error; clc;
warning off; close all;

T=Fun_open_calib('000089.txt','./');%标定文件
ima	= './000089.png';    %彩色图
ImaRGB=imread(ima);
fd = fopen(sprintf('./000089.bin'),'rb');
velo = fread(fd,[4 inf],'single')';
fclose(fd);

idx = velo(:,1)<0;
velo(idx,:) = [];

px = (T.P2 * T.R0_rect * T.Tr_velo_to_cam * velo')';% 旋转到RGB平面，公式参考KITTI官网关于校正文件的使用的readme
px(:,1) = px(:,1)./px(:,3);
px(:,2) = px(:,2)./px(:,3);

ix = px(:,1)<1;                 px(ix,:)=[];velo(ix,:)=[];% 忽略位置小于1的像素
ix = px(:,1)>size(ImaRGB,2);    px(ix,:)=[];velo(ix,:)=[];% 忽略超出RGB范围的像素   

ppx=px;

Pts = zeros(size(px,1),4);
Pts = sortrows(px,2);

c_px = floor(min(px(:,2)));% 对第二列最小的数向下取证，即最小的像素位置
i_size = size(ImaRGB(c_px:end,:,1));% 
Ima3D = zeros( size(ImaRGB(:,:,1)) );% 初始化一个矩阵用来存储稠密化的深度图
Ima3D(c_px:end,:) = fun_dense3D(Pts,[c_px i_size]); 
idxx=find(Ima3D==0); % find all 1.5
Ima3D(idxx)=max(max(Ima3D)); % set 1 to these indexes
Ima_Range = uint8( 255*Ima3D/max(max(Ima3D)));

figure(1);
subplot(2,1,1);
image(ImaRGB);
coord=fix(Pts(:,1:3));
collor=[Pts(:,1)/max(Pts(:,1)) Pts(:,2)/max(Pts(:,2)) Pts(:,3)/max(Pts(:,3))];

%hsv = rgb2hsv(ImaRGB);
%gray=rgb2gray(hsv);
%gray=imbinarize(hsv);

hold on;
scatter(coord(:,1),coord(:,2),1,collor);
%imwrite(ImaRGB,'./000089_rgb+raw_lidar.png');
subplot(2,1,2);
image(Ima_Range);

%imwrite(Ima_Range,'./000089_upsample.png');

% figure(2);
% subplot(2,1,1);
% image(hsv);
% subplot(2,1,2);
% image(hsv);
% image(gray(:,3));