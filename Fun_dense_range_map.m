function Ima_Range = Fun_dense_range_map(calib,calib_dir,base_dir,...
    frame,ImaRGB)

T = Fun_open_calib(calib.name,calib_dir);
% T = Fun_open_calib(calib(frame+1).name,calib_dir);%
% 这句是我加的，为的是可以单独这里的代码，读取第frame个雷达文件

fd = fopen(sprintf('%s/%06d.bin',base_dir,frame),'rb');% 打开雷达二进制文件
if fd < 1
    fprintf('No LIDAR files !!!\n');
    keyboard
else
velo = fread(fd,[4 inf],'single')';% 读取雷达数据，存为n*4的矩阵
%figure;plot3(velo(:,1),velo(:,2),velo(:,3),'.');xlabel('X');ylabel('Y');zlabel('Z');title('Original Velo');
% 这里画图的程序是我加的，为了看中间结果
fclose(fd);
end

% remove all points behind image plane (approximation)
idx = velo(:,1)<0;% 忽略第一列中小于5的点，为什么取值为5，我也不知道，why 5?
velo(idx,:) = [];
% figure;plot3(px(:,1),px(:,2),px(:,3),'.');% 这里画图的程序是我加的，为了看中间结果
%ppx=velo;
% project to image plane (exclude luminance)
px = (T.P2 * T.R0_rect * T.Tr_velo_to_cam * velo')';% 旋转到RGB平面，公式参考KITTI官网关于校正文件的使用的readme
px(:,1) = px(:,1)./px(:,3);
px(:,2) = px(:,2)./px(:,3);
ppx=px;
%figure;plot3(px(:,1),px(:,2),px(:,3),'.');xlabel('X');ylabel('Y');zlabel('Z');title('project to image plane');
% 这里画图的程序是我加的，为了看中间结果

% % -----------------------------------------------------------------------
ix = px(:,1)<1;                 px(ix,:)=[];velo(ix,:)=[];% 忽略位置小于1的像素
ix = px(:,1)>size(ImaRGB,2);    px(ix,:)=[];velo(ix,:)=[];% 忽略超出RGB范围的像素   
%  figure;% 这里画图的程序是我加的，为了看中间结果
%  m3=max(px(:,3));collor=[px(:,3)/m3 px(:,3)/m3 px(:,3)/m3];
%  for i=1:length(collor)
%      hold on;
%      plot3(px(i,1),px(i,2),px(i,3),'.','color',collor(i,:));
%  end
%  xlabel('X');ylabel('Y');zlabel('Z');
% % Ordering
Pts = zeros(size(px,1),4);
Pts = sortrows(px,2);% 按照第二列排序
% % ======================= Interpolation / Upsampling :::
c_px = floor(min(px(:,2)));% 对第二列最小的数向下取证，即最小的像素位置
i_size = size(ImaRGB(c_px:end,:,1));% 
Ima3D = zeros( size(ImaRGB(:,:,1)) );% 初始化一个矩阵用来存储稠密化的深度图

% Simply type: mex fun_dense3D.cpp
Ima3D(c_px:end,:) = fun_dense3D(Pts,[c_px i_size]); % MEX-file 调用C++程序
% % -----------------------------------------------------------------------
% Normalization 8 bits

idxx=find(Ima3D==0); % find all 1.5
Ima3D(idxx)=max(max(Ima3D)); % set 1 to these indexes
Ima_Range = uint8( 255*Ima3D/max(max(Ima3D)) ); % :)% 转为uint8格式，得到最终稠密化的深度图
%显示雷达散点图到image
figure(1);
%subplot(2,1,1);
image(ImaRGB);
coord=fix(Pts(:,1:3));
%m3=max(ppx(:,3));
%collor=[ppx(:,3)/max(ppx(:,3)) ppx(:,3)/max(ppx(:,3)) ppx(:,3)/max(ppx(:,3)) ppx(:,3)/max(ppx(:,3))];
%coordd=[sqrt(coord(:,1).*coord(:,1)+coord(:,2).*coord(:,2)) sqrt(coord(:,1).*coord(:,1)+coord(:,2).*coord(:,2)) sqrt(coord(:,1).*coord(:,1)+coord(:,2).*coord(:,2))];
velo=abs(velo);
collor=[velo(:,1)/max(velo(:,1)) velo(:,2)/max(velo(:,2)) velo(:,3)/max(velo(:,3))];
hold on;
%scatter(coord(:,1),coord(:,2),1,'red');
%subplot(2,1,1);
scatter(coord(:,1),coord(:,2),3,collor);

end %END main Function
